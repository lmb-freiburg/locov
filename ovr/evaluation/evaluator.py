import os
import time
import logging
import datetime
import itertools
import numpy as np
from collections import OrderedDict, abc
from timeit import default_timer as timer
from typing import Callable, Dict, Tuple, List, Union
from contextlib import ExitStack, contextmanager

import detectron2.utils.comm as comm
from detectron2.evaluation.evaluator import inference_on_dataset, DatasetEvaluator
from detectron2.utils.comm import get_world_size
from detectron2.utils.logger import log_every_n_seconds
from detectron2.evaluation import COCOEvaluator, DatasetEvaluators
from detectron2.evaluation.lvis_evaluation import LVISEvaluator

import torch
from torch.nn.parallel import DistributedDataParallel

from ovr.misc import dot_similarity_np, l2_normalize_np


def select_and_build_evaluator(cfg, dataset_name, output_folder=None):
    """
    Based on detectron build_evaluator function in trainer
    """
    # Get evaluation type
    if cfg.MODEL.META_ARCHITECTURE in {
        "MMSSGridModel",
        "DistillMMSSGridModel",
        "DistillMMSSMixTokensGridModel",
        "HierarchicalDistillMMSSGridModel",
    }:
        # No proposals of bounding boxes to evaluate detection
        evaluator_type = "ovr"
        return evaluator_type, None
    elif "lvis" in dataset_name:
        evaluator_type = "lvis"
    else:
        evaluator_type = "coco"

    if output_folder is None:
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
    evaluator_list = []
    if evaluator_type == "coco":
        evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder))
    if evaluator_type == "lvis":
        evaluator_list.append(LVISEvaluator(dataset_name, output_dir=output_folder))
    if cfg.MODEL.META_ARCHITECTURE in {
        "DistillProposalMMSSRCNN",
        "DistillProposalMMSSMixTokensRCNN",
        "DistillOnlyProposalMMSSRCNN",
        "HierarchicalDistillProposalMMSSRCNN",
    }:
        evaluator_type = "loss_and_" + evaluator_type

    if len(evaluator_list) == 0:
        raise NotImplementedError(
            "no Evaluator for the dataset {} with the type {}".format(
                dataset_name, evaluator_type
            )
        )
    elif len(evaluator_list) == 1:
        return evaluator_type, evaluator_list[0]
    return evaluator_type, DatasetEvaluators(evaluator_list)


def inference_on_dataset_evaluation_type(model, data_loader, evaluator, evaluator_type):
    """
    Influenced from detectron2's own inference_on_dataset but used to do inference depending on the evaluation type
    """
    if evaluator_type == "coco":
        results = inference_on_dataset(model, data_loader, evaluator)
        return results
    elif evaluator_type == "lvis":
        results = inference_on_dataset(model, data_loader, evaluator)
        return results
    elif evaluator_type == "ovr":
        results = inference_on_caption_ovr_dataset(model, data_loader)
        return results
    elif evaluator_type in {"loss_and_coco", "loss_and_lvis"}:
        ovr_results = inference_on_caption_ovr_dataset(
            model, data_loader[0], change_training_mode=False
        )
        coco_results = inference_on_dataset(model, data_loader[1], evaluator)
        results = {}
        for res_dict in ovr_results:
            results.update(res_dict)
        results.update(coco_results)
        return results
    else:
        assert evaluator_type == "mitstates", "Evaluator type not define: {}".format(
            evaluator_type
        )


def inference_on_caption_ovr_dataset(model, data_loader, change_training_mode=True):
    """

    :param model(callable): a callable which takes an object from
            `data_loader` and returns some outputs.

            If it's an nn.Module, it will be temporarily set to `eval` mode.
            If you wish to evaluate a model in `training` mode instead, you can
            wrap the given model and override its behavior of `.eval()` and `.train()`.
    :param data_loader: an iterable object with a length.
            The elements it generates will be the inputs to the model.
    :param evaluator (DatasetEvaluator): the evaluator to run. Use `None` if you only want
            to benchmark, but don't want to do any evaluation.
    :return:
    """
    num_devices = get_world_size()
    logger = logging.getLogger(__name__)
    logger.info("Start inference on {} images".format(len(data_loader)))

    total_loss_dict = {}
    total_metrics_dict = {}
    total = len(data_loader)  # inference data loader must have a fixed length
    num_warmup = min(5, total - 1)
    start_time = time.perf_counter()
    total_compute_time = 0

    training_mode = model.training
    if change_training_mode:
        model.eval()
    with torch.no_grad():
        for idx, inputs in enumerate(data_loader):
            start_compute_time = time.perf_counter()
            out_dict, loss_dict = model(inputs)
            loss_dict["Total Loss"] = sum(loss_dict.values())

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            total_compute_time += time.perf_counter() - start_compute_time

            for k, v in loss_dict.items():
                if k not in total_loss_dict.keys():
                    total_loss_dict[k] = 0
                total_loss_dict[k] += v.detach().cpu()

            for k, v in out_dict.items():
                if k not in total_metrics_dict.keys():
                    total_metrics_dict[k] = 0
                total_metrics_dict[k] += v.detach().cpu()

            iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
            seconds_per_img = total_compute_time / iters_after_start
            if idx >= num_warmup * 2 or seconds_per_img > 5:
                total_seconds_per_img = (
                    time.perf_counter() - start_time
                ) / iters_after_start
                eta = datetime.timedelta(
                    seconds=int(total_seconds_per_img * (total - idx - 1))
                )
                log_every_n_seconds(
                    logging.INFO,
                    "Inference done {}/{}. {:.4f} s / img. ETA={}".format(
                        idx + 1, total, seconds_per_img, str(eta)
                    ),
                    n=5,
                )
            # TODO: remove this break
            # if idx>500:
            #     break

    # Measure the time only for this worker (before the synchronization barrier)
    total_time = time.perf_counter() - start_time
    total_time_str = str(datetime.timedelta(seconds=total_time))
    # NOTE this format is parsed by grep
    logger.info(
        "Total inference time: {} ({:.6f} s / img per device, on {} devices)".format(
            total_time_str, total_time / (total - num_warmup), num_devices
        )
    )
    total_compute_time_str = str(datetime.timedelta(seconds=int(total_compute_time)))
    logger.info(
        "Total inference pure compute time: {} ({:.6f} s / img per device, on {} devices)".format(
            total_compute_time_str,
            total_compute_time / (total - num_warmup),
            num_devices,
        )
    )

    validation_loss_dict = {}
    for k, v in total_loss_dict.items():
        # validation_loss_dict['Val '+k] = v/len(data_loader)
        validation_loss_dict[k] = v / len(data_loader)
    validation_metrics_dict = {}
    for k, v in total_metrics_dict.items():
        # validation_metrics_dict['Val '+k] = v/len(data_loader)
        validation_metrics_dict[k] = v / len(data_loader)

    model.train(training_mode)
    return validation_metrics_dict, validation_loss_dict
