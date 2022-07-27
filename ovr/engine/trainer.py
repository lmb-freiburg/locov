import os
import time
import logging
import torch
import json
from typing import Dict, List, Tuple, Union
from torch.nn.parallel import DistributedDataParallel
from fvcore.nn.precise_bn import get_bn_modules
import numpy as np
from collections import OrderedDict
import GPUtil

import detectron2.utils.comm as comm
from detectron2.engine import DefaultTrainer, TrainerBase
from detectron2.utils.events import (
    EventStorage,
    get_event_storage,
    JSONWriter,
    TensorboardXWriter,
)
from detectron2.evaluation import print_csv_format, verify_results
from detectron2.data import MetadataCatalog, build_detection_train_loader
from detectron2.engine import hooks
from detectron2.evaluation import verify_results
from detectron2.evaluation.testing import flatten_results_dict

from ovr.evaluation.evaluator import inference_on_dataset_evaluation_type
from ovr.data.mappers import get_mapper
from ovr.utils.misc import CalcWriter, AllMetricPrinter, count_parameters
from ovr.utils.events import OvrMetricPrinter
from ovr.engine.solver import build_optimizer
from ovr.data.dataloader import build_detection_test_loader, build_detection_val_loader
from ovr.utils.checkpoint import WSOGCheckpointer
from ovr.evaluation.evaluator import select_and_build_evaluator


class OVRTrainer(DefaultTrainer):
    def __init__(self, cfg):
        """
        Args:
            cfg (CfgNode):
        """
        super(OVRTrainer).__init__()
        logger = logging.getLogger("detectron2")
        cfg = DefaultTrainer.auto_scale_workers(cfg, comm.get_world_size())

        # Assume these objects must be constructed in this order.
        model = self.build_model(cfg)
        optimizer = self.build_optimizer(cfg, model)
        data_loader = self.build_train_loader(cfg)

        # Load noun embeddings
        dataset_name = cfg.DATASETS.TRAIN[0]
        if cfg.MODEL.ROI_BOX_HEAD.EMBEDDING_BASED:
            self.load_embeddings(cfg, dataset_name, model)
        # Check which parameters require gradient
        logger.info("Parameters who need back-prop gradients")
        count_parameters(model, logger=logger)

        # For training, wrap with DDP. But don't need this for inference.
        if comm.get_world_size() > 1:
            # TODO: when using a new model track unused parameters
            # model = DistributedDataParallel(model, device_ids=[comm.get_local_rank()], broadcast_buffers=False, find_unused_parameters=True)
            model = DistributedDataParallel(
                model, device_ids=[comm.get_local_rank()], broadcast_buffers=False
            )
        self._trainer = SimpleTrainerMMSS(model, data_loader, optimizer)

        self.scheduler = self.build_lr_scheduler(cfg, optimizer)
        # Assume no other objects need to be checkpointed.
        # We can later make it checkpoint the stateful hooks
        self.checkpointer = WSOGCheckpointer(
            # Assume you want to save checkpoints together with logs/statistics
            model,
            cfg.OUTPUT_DIR,
            optimizer=optimizer,
            scheduler=self.scheduler,
        )
        self.start_iter = 0
        self.max_iter = cfg.SOLVER.MAX_ITER
        self.cfg = cfg
        self.iter = 0

        # clean hooks and register
        self._hooks = []
        self.register_hooks(self.build_hooks())

        self.best_metric = (cfg.TEST.SAVE_MODEL_BEST_METRIC, -1)

    def train_loop(self, start_iter: int, max_iter: int):
        """
        Args:
            start_iter, max_iter (int): See docs above
        """
        logger = logging.getLogger(__name__)
        logger.info("Starting training from iteration {}".format(start_iter))

        self.iter = self.start_iter = start_iter
        self.max_iter = max_iter

        with EventStorage(start_iter) as self.storage:
            try:
                self.before_train()
                if self.cfg.TEST.EVAL_INIT and self.start_iter == 0:
                    for h in self._hooks:
                        if isinstance(h, hooks.EvalHook):
                            h._do_eval()
                for self.iter in range(start_iter, max_iter):
                    self.before_step()
                    self.run_step()
                    if self.iter % 100 == 0 and 0 < self.iter < 300:
                        GPUtil.showUtilization(all=True)
                    self.after_step()  # Here it does evaluation
            except Exception:
                logger.exception("Exception during training:")
                raise
            finally:
                self.after_train()

    def train(self):
        self.train_loop(self.start_iter, self.max_iter)
        if hasattr(self, "_last_eval_results") and comm.is_main_process():
            verify_results(self.cfg, self._last_eval_results)
            return self._last_eval_results

    @classmethod
    def build_train_loader(cls, cfg):
        """
        Returns:
            iterable

        It now calls :func:`detectron2.data.build_detection_train_loader`.
        Overwrite it if you'd like a different data loader.
        """
        for train_set in cfg.DATASETS.TRAIN:
            # if cfg.INPUT.NOISE_OFFLINE:
            #     fix_noise_offline(cfg, train_set)
            mapper = get_mapper(train_set, cfg, True)
        return build_detection_train_loader(cfg, mapper=mapper)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        """
        Returns:
            iterable

        It now calls :func:`detectron2.data.build_detection_test_loader`.
        Overwrite it if you'd like a different data loader.
        """
        mapper = get_mapper(dataset_name, cfg, False)
        return build_detection_test_loader(cfg, dataset_name, mapper=mapper)

    @classmethod
    def build_val_loader(cls, cfg, dataset_name):
        """
        Returns:
            iterable

        It now calls :func:`detectron2.data.build_detection_test_loader`.
        Overwrite it if you'd like a different data loader.
        """
        mapper = get_mapper(dataset_name, cfg, False)
        return build_detection_val_loader(cfg, dataset_name, mapper=mapper)
        # return build_detection_train_loader(dataset_name, mapper=mapper, sampler=None, total_batch_size=int((cfg.SOLVER.IMS_PER_BATCH+1)/2))

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        return select_and_build_evaluator(cfg, dataset_name, output_folder)

    @classmethod
    def test(cls, cfg, model, evaluators=None):
        """
        Args:
            cfg (CfgNode):
            model (nn.Module):
            evaluators (list[DatasetEvaluator] or None): if None, will call
                :meth:`build_evaluator`. Otherwise, must have the same length as
                ``cfg.DATASETS.TEST``.

        Returns:
            dict: a dict of result metrics
        """
        torch.cuda.empty_cache()
        logger = logging.getLogger(__name__)
        results = OrderedDict()
        for idx, dataset_name in enumerate(cfg.DATASETS.TEST):
            data_loader = cls.build_test_loader(cfg, dataset_name)
            evaluator_type, evaluator = cls.build_evaluator(cfg, dataset_name)
            print("evaluator_type", evaluator_type)

            cls.load_embeddings(cfg, dataset_name, model)
            if cfg.TEST.DO_EVAL:
                val_data_loader = cls.build_val_loader(cfg, dataset_name)
                # validation_loss = inference_on_caption_ovr_dataset(model, val_data_loader, change_to_eval_mode=False)
                data_loader = [val_data_loader, data_loader]
            validation_results = inference_on_dataset_evaluation_type(
                model, data_loader, evaluator, evaluator_type
            )
            if isinstance(validation_results, tuple):
                validation_metrics_dict, validation_loss_dict = validation_results
            else:
                validation_loss_dict = validation_results
                validation_metrics_dict = {}
            results[dataset_name] = OrderedDict()
            results[dataset_name].update(validation_metrics_dict)
            results[dataset_name].update(validation_loss_dict)
            if comm.is_main_process():
                assert isinstance(
                    results[dataset_name], dict
                ), "Evaluator must return a dict on the main process. Got {} instead.".format(
                    results[dataset_name]
                )
                logger.info(
                    "Evaluation results for {} in csv format:".format(dataset_name)
                )
                print_csv_format(results[dataset_name])

        return results

    def build_hooks(self):
        """
        Build a list of default hooks, including timing, evaluation,
        checkpointing, lr scheduling, precise BN, writing events.

        Returns:
            list[HookBase]:
        """
        cfg = self.cfg.clone()
        cfg.defrost()
        cfg.DATALOADER.NUM_WORKERS = 0

        ret = [
            hooks.IterationTimer(),
            hooks.LRScheduler(self.optimizer, self.scheduler),
            hooks.PreciseBN(
                cfg.TEST.EVAL_PERIOD,
                self.model,
                self.build_train_loader(cfg),
                cfg.TEST.PRECISE_BN.NUM_ITER,
            )
            if cfg.TEST.PRECISE_BN.ENABLED and get_bn_modules(self.model)
            else None,
        ]

        if comm.is_main_process():
            ret.append(
                hooks.PeriodicCheckpointer(
                    self.checkpointer,
                    cfg.SOLVER.CHECKPOINT_PERIOD,
                    max_to_keep=2,
                )
            )

        def test_and_save_results():
            self._last_eval_results = self.test(self.cfg, self.model)
            dataset_name = self.cfg.DATASETS.TRAIN[0]
            self.load_embeddings(self.cfg, dataset_name, self.model)
            if comm.is_main_process():
                flatten_results = flatten_results_dict(self._last_eval_results)
                self.best_metric = self.checkpointer.save_best_metric(
                    "model_best", flatten_results, self.best_metric, self.iter
                )
            return self._last_eval_results

        ret.append(hooks.EvalHook(cfg.TEST.EVAL_PERIOD, test_and_save_results))

        if comm.is_main_process():
            ret.append(
                hooks.PeriodicWriter(
                    [
                        OvrMetricPrinter(self.max_iter),
                        AllMetricPrinter(self.max_iter),
                        JSONWriter(os.path.join(self.cfg.OUTPUT_DIR, "metrics.json")),
                        TensorboardXWriter(self.cfg.OUTPUT_DIR),
                    ],
                    period=cfg.SOLVER.LOG_PERIOD,
                )
            )
            # ret.append(hooks.PeriodicWriter(self.build_writers(), period=cfg.SOLVER.LOG_PERIOD))
            ret.append(
                hooks.PeriodicWriter(
                    [
                        CalcWriter(
                            os.path.join(self.cfg.OUTPUT_DIR, "metrics_log.csv"),
                            self.cfg.SOLVER.EPOCH_ITER_SIZE,
                        )
                    ],
                    period=self.cfg.SOLVER.EPOCH_ITER_SIZE // 4,
                )
            )
        return ret

    def resume_or_load(self, resume=True):
        """
        If `resume==True` and `cfg.OUTPUT_DIR` contains the last checkpoint (defined by
        a `last_checkpoint` file), resume from the file. Resuming means loading all
        available states (eg. optimizer and scheduler) and update iteration counter
        from the checkpoint. ``cfg.MODEL.WEIGHTS`` will not be used.

        Otherwise, this is considered as an independent training. The method will load model
        weights from the file `cfg.MODEL.WEIGHTS` (but will not load other states) and start
        from iteration 0.

        Args:
            resume (bool): whether to do resume or not
        """
        # self.checkpointer.resume_or_load(self.cfg.MODEL.WEIGHTS, resume=resume)
        rename_keys = {}
        if self.cfg.MODEL.LOAD_EMB_PRED_FROM_MMSS_HEAD:
            # rename_keys = {"backbone.res5":'roi_heads.res5', "mmss_heads.GroundingHead.v2l_projection":"roi_heads.box_predictor.emb_pred"}
            # dictionary with {olk_key_name : [new_key_names]}
            rename_keys = {
                "backbone.res5": [
                    "roi_heads.res5",
                    "roi_heads.res5_head1",
                    "roi_heads.res5_head2",
                ],
                "roi_heads.res5": [
                    "backbone.res5",
                    "roi_heads.res5_head1",
                    "roi_heads.res5_head2",
                ],
                "roi_heads.res5_head1": ["backbone.res5", "roi_heads.res5"],
                "res5": [
                    "backbone.res5",
                    "roi_heads.res5",
                    "roi_heads.res5_head1",
                    "roi_heads.res5_head2",
                ],
                "mmss_heads.GroundingHead.v2l_projection": [
                    "roi_heads.box_predictor.emb_pred",
                    "roi_heads.emb_pred",
                ],
                "roi_heads.box_predictor.emb_pred": ["roi_heads.emb_pred"],
            }
        self.checkpointer.resume_or_load_renaming_keys(
            self.cfg.MODEL.WEIGHTS, resume=resume, rename_keys=rename_keys
        )
        if self.cfg.MODEL.PROJECTION_WEIGHTS != "":
            rename_keys = {
                "mmss_heads.GroundingHead.v2l_projection": [
                    "roi_heads.box_predictor.emb_pred",
                    "roi_heads.emb_pred",
                ],
                "roi_heads.box_predictor.emb_pred": ["roi_heads.emb_pred"],
            }
            self.checkpointer.load_projection_layer(
                self.cfg.MODEL.PROJECTION_WEIGHTS,
                resume=resume,
                rename_keys=rename_keys,
            )
        if resume and self.checkpointer.has_checkpoint():
            # The checkpoint stores the training iteration that just finished, thus we start
            # at the next iteration
            load_iter = (
                self.checkpointer.get_checkpoint_file()
                .split("/")[-1]
                .split(".")[0]
                .split("_")[-1]
            )
            if load_iter == "best":
                best_dict = json.load(
                    open(
                        self.checkpointer.get_checkpoint_file().replace(
                            ".pth", ".json"
                        ),
                        "r",
                    )
                )
                load_iter = best_dict["iteration"]
            self.iter = int(load_iter)
            self.start_iter = self.iter + 1

    @classmethod
    def load_embeddings(cls, cfg, dataset_name, model):
        logger = logging.getLogger(__name__)

        # Load noun embeddings
        dataset_metadata = MetadataCatalog.get(dataset_name)
        if hasattr(dataset_metadata, "class_embeddings"):
            cls_emb = dataset_metadata.class_embeddings
        elif hasattr(dataset_metadata, "class_emb_mtx"):
            cls_emb = dataset_metadata.class_emb_mtx
        else:
            raise NotImplementedError("Not class embeddings")

        module = (
            model.module
            if isinstance(model, torch.nn.parallel.DistributedDataParallel)
            else model
        )
        if hasattr(module, "roi_heads") and cfg.MODEL.ROI_BOX_HEAD.EMBEDDING_BASED:
            if hasattr(module.roi_heads, "box_predictor"):
                if module.roi_heads.box_predictor.embedding_based:
                    if logger is not None:
                        logger.info(
                            "Setting embeddings for noun detection: " + dataset_name
                        )
                    else:
                        print("Setting embeddings for noun detection: " + dataset_name)
                    module.roi_heads.box_predictor.set_class_embeddings(cls_emb)
                    if hasattr(module.roi_heads, "num_classes"):
                        module.roi_heads.num_classes = (
                            module.roi_heads.box_predictor.num_classes
                        )
            elif hasattr(module.roi_heads, "embedding_based"):
                if module.roi_heads.embedding_based:
                    if logger is not None:
                        logger.info(
                            "Setting embeddings for noun detection: " + dataset_name
                        )
                    else:
                        print("Setting embeddings for noun detection: " + dataset_name)
                    module.roi_heads.set_class_embeddings(cls_emb)
                    if hasattr(module, "num_classes"):
                        module.num_classes = module.roi_heads.num_classes

    def extract_embeddings(self, cfg, model):
        state_dict = {}
        if cfg.MODEL.ROI_BOX_HEAD.EMBEDDING_BASED:
            module = (
                model.module
                if isinstance(model, torch.nn.parallel.DistributedDataParallel)
                else model
            )
            if hasattr(module, "roi_heads"):
                if hasattr(module.roi_heads, "box_predictor"):
                    state_dict = module.roi_heads.box_predictor.state_dict()
                elif hasattr(module.roi_heads, "embedding_based"):
                    state_dict = module.roi_heads.state_dict()
        return state_dict

    @classmethod
    def build_optimizer(cls, cfg, model):
        """
        Returns:
            torch.optim.Optimizer:

        It now calls :func:`detectron2.solver.build_optimizer`.
        Overwrite it if you'd like a different optimizer.
        """
        return build_optimizer(cfg, model)


class SimpleTrainerMMSS(TrainerBase):
    """
    A simple trainer for the most common type of task:
    single-cost single-optimizer single-data-source iterative optimization,
    optionally using data-parallelism.
    It assumes that every step, you:

    1. Compute the loss with a data from the data_loader.
    2. Compute the gradients with the above loss.
    3. Update the model with the optimizer.

    All other tasks during training (checkpointing, logging, evaluation, LR schedule)
    are maintained by hooks, which can be registered by :meth:`TrainerBase.register_hooks`.

    If you want to do anything fancier than this,
    either subclass TrainerBase and implement your own `run_step`,
    or write your own training loop.
    """

    def __init__(self, model, data_loader, optimizer):
        """
        Args:
            model: a torch Module. Takes a data from data_loader and returns a
                dict of losses.
            data_loader: an iterable. Contains data to be used to call model.
            optimizer: a torch optimizer.
        """
        super().__init__()

        """
        We set the model to training mode in the trainer.
        However it's valid to train a model that's in eval mode.
        If you want your model (or a submodule of it) to behave
        like evaluation during training, you can overwrite its train() method.
        """
        model.train()

        self.model = model
        self.data_loader = data_loader
        self._data_loader_iter = iter(data_loader)
        self.optimizer = optimizer

    def run_step(self):
        """
        Implement the standard training logic described above.
        """
        assert self.model.training, "[SimpleTrainer] model was changed to eval mode!"
        start = time.perf_counter()
        """
        If you want to do something with the data, you can wrap the dataloader.
        """
        data = next(self._data_loader_iter)
        data_time = time.perf_counter() - start

        """
        If you want to do something with the losses, you can wrap the model.
        """
        result_model = self.model(data)

        if isinstance(result_model, tuple):
            out_dict, loss_dict = result_model
        else:
            loss_dict = result_model
            out_dict = {}
        losses = sum(loss_dict.values())

        """
        If you need to accumulate gradients or do something similar, you can
        wrap the optimizer with your custom `zero_grad()` method.
        """
        self.optimizer.zero_grad()
        losses.backward()

        self._write_metrics(loss_dict, data_time)
        self._write_metrics(out_dict, data_time, calc_total=False)

        """
        If you need gradient clipping/scaling or other processing, you can
        wrap the optimizer with your custom `step()` method. But it is
        suboptimal as explained in https://arxiv.org/abs/2006.15704 Sec 3.2.4
        """
        self.optimizer.step()

    def _write_metrics(
        self,
        metrics_dict: Dict[str, torch.Tensor],
        data_time: float,
        prefix: str = "",
        calc_total: bool = True,
        return_metrics: bool = False,
    ):
        """
        Args:
            metrics_dict (dict): dict of scalar losses
            data_time (float): time taken by the dataloader iteration
        """
        metrics_dict = {k: v.detach().cpu().item() for k, v in metrics_dict.items()}
        metrics_dict["data_time"] = data_time

        # Gather metrics among all workers for logging
        # This assumes we do DDP-style training, which is currently the only
        # supported method in detectron2.
        all_metrics_dict = comm.gather(metrics_dict)

        if comm.is_main_process():
            storage = get_event_storage()

            # data_time among workers can have high variance. The actual latency
            # caused by data_time is the maximum among workers.
            data_time = np.max([x.pop("data_time") for x in all_metrics_dict])
            storage.put_scalar("data_time", data_time)

            # average the rest metrics
            metrics_dict = {
                k: np.mean([x[k] for x in all_metrics_dict])
                for k in all_metrics_dict[0].keys()
            }
            if calc_total:
                total_losses_reduced = sum(metrics_dict.values())
                if not np.isfinite(total_losses_reduced):
                    raise FloatingPointError(
                        f"Loss became infinite or NaN at iteration={self.iter}!\n"
                        f"loss_dict = {metrics_dict}"
                    )

                storage.put_scalar("{}total_loss".format(prefix), total_losses_reduced)
            if len(metrics_dict) > 1:
                storage.put_scalars(**metrics_dict)

        if return_metrics:
            return all_metrics_dict
