import itertools
from enum import Enum
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Type, Union
import torch
from detectron2.config import CfgNode
from detectron2.solver.build import maybe_add_gradient_clipping


def build_optimizer(
    cfg: CfgNode,
    model: torch.nn.Module,
    overrides_dict: Optional[Dict[str, Dict[str, float]]] = None,
) -> torch.optim.Optimizer:
    """
    Build an optimizer from config.
    """
    params = get_default_optimizer_params(
        model,
        base_lr=cfg.SOLVER.BASE_LR,
        weight_decay=cfg.SOLVER.WEIGHT_DECAY,
        weight_decay_norm=cfg.SOLVER.WEIGHT_DECAY_NORM,
        bias_lr_factor=cfg.SOLVER.BIAS_LR_FACTOR,
        weight_decay_bias=cfg.SOLVER.WEIGHT_DECAY_BIAS,
        overrides=overrides_dict,
    )

    return maybe_add_gradient_clipping(cfg, torch.optim.SGD)(
        params,
        cfg.SOLVER.BASE_LR,
        momentum=cfg.SOLVER.MOMENTUM,
        nesterov=cfg.SOLVER.NESTEROV,
    )


def get_default_optimizer_params(
    model: torch.nn.Module,
    base_lr,
    weight_decay,
    weight_decay_norm,
    bias_lr_factor=1.0,
    weight_decay_bias=None,
    overrides: Optional[Dict[str, Dict[str, float]]] = None,
):
    """
    Get default param list for optimizer

    Args:
        overrides (dict: str -> (dict: str -> float)):
            if not `None`, provides values for optimizer hyperparameters
            (LR, weight decay) for module parameters with a given name; e.g.
            {"embedding": {"lr": 0.01, "weight_decay": 0.1}} will set the LR and
            weight decay values for all module parameters named `embedding` (default: None)
    """
    if weight_decay_bias is None:
        weight_decay_bias = weight_decay
    norm_module_types = (
        torch.nn.BatchNorm1d,
        torch.nn.BatchNorm2d,
        torch.nn.BatchNorm3d,
        torch.nn.SyncBatchNorm,
        # NaiveSyncBatchNorm inherits from BatchNorm2d
        torch.nn.GroupNorm,
        torch.nn.InstanceNorm1d,
        torch.nn.InstanceNorm2d,
        torch.nn.InstanceNorm3d,
        torch.nn.LayerNorm,
        torch.nn.LocalResponseNorm,
    )
    params: List[Dict[str, Any]] = []
    memo: Set[torch.nn.parameter.Parameter] = set()
    for module_name, module in model.named_modules():
        for module_param_name, value in module.named_parameters(recurse=False):
            if not value.requires_grad:
                continue
            # Avoid duplicating parameters
            if value in memo:
                continue
            memo.add(value)

            schedule_params = {
                "lr": base_lr,
                "weight_decay": weight_decay,
            }
            if isinstance(module, norm_module_types):
                schedule_params["weight_decay"] = weight_decay_norm
            elif module_param_name == "bias":
                # NOTE: unlike Detectron v1, we now default BIAS_LR_FACTOR to 1.0
                # and WEIGHT_DECAY_BIAS to WEIGHT_DECAY so that bias optimizer
                # hyperparameters are by default exactly the same as for regular
                # weights.
                schedule_params["lr"] = base_lr * bias_lr_factor
                schedule_params["weight_decay"] = weight_decay_bias
            if overrides is not None:
                if module_param_name in overrides:
                    schedule_params.update(overrides[module_param_name])
                for override_name in overrides.keys():
                    if override_name in module_name:
                        schedule_params.update(overrides[override_name])

            params += [
                {
                    "params": [value],
                    "lr": schedule_params["lr"],
                    "weight_decay": schedule_params["weight_decay"],
                }
            ]

    return params
