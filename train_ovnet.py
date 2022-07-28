"""
This code is based on the Detectron2 repository.
For usage, see the License of Detectron2 under:
https://github.com/facebookresearch/detectron2

Command for usage
python train_ovnet.py --resume --num-gpus 8 --config-file configs/coco_lsm.yaml
"""

import logging
from collections import OrderedDict
import torch
from prettytable import PrettyTable
from ast import literal_eval

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.engine import default_argument_parser, default_setup, launch
from detectron2.evaluation import verify_results
from detectron2.config import get_cfg

from ovr.engine.trainer import OVRTrainer as Trainer
from ovr.data.register_datasets import get_register_dataset
from ovr.config.config_utils import edit_output_dir_exp_specific
from ovr.config.config import add_ovr_config


def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params += param
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_ovr_config(cfg)
    cfg.merge_from_file(args.config_file)
    literal_ops = []
    for x in args.opts:
        try:
            literal_ops.append(literal_eval(x))
        except (SyntaxError, ValueError):
            literal_ops.append(x)

    cfg.merge_from_list(literal_ops)
    cfg = edit_output_dir_exp_specific(cfg)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)

    if args.eval_only:
        for test_set in cfg.DATASETS.TEST:
            register_dataset = get_register_dataset(test_set)
            register_dataset(test_set)
            if "coco" not in cfg.DATASETS.TRAIN[0] and "coco" in test_set:
                register_dataset = get_register_dataset(cfg.DATASETS.TRAIN[0])
                register_dataset(cfg.DATASETS.TRAIN[0])
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    for train_set in cfg.DATASETS.TRAIN:
        register_dataset = get_register_dataset(train_set)
        register_dataset(train_set)
    if cfg.TEST.EVAL_PERIOD > 0:
        for test_set in cfg.DATASETS.TEST:
            register_dataset = get_register_dataset(test_set)
            register_dataset(test_set)

    trainer = Trainer(cfg)
    # count_parameters(trainer.model)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()

    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
