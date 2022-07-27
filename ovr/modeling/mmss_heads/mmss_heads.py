"""
Edited from https://github.com/alirezazareian/ovr-cnn/maskrcnn_benchmark/modeling/mmss_heads/__init__.py
"""
import torch
from torch import nn
from ovr.modeling.mmss_heads.grounding_head import (
    MMSS_HEADS_REGISTRY,
    build_grounding_head,
)
from ovr.modeling.mmss_heads.transformer_head import build_transformer_head
from ovr.modeling.mmss_heads.mlp_head import build_mlp_head


def build_mmss_heads(cfg, *args, **kwargs):
    heads = {}
    for head_type in cfg.MODEL.MMSS_HEAD.TYPES:
        assert (
            head_type in MMSS_HEADS_REGISTRY
        ), "cfg.MODEL.MMSS_HEAD.TYPE: {} is not registered in Registry".format(
            head_type
        )
        if head_type == "GroundingHead":
            heads[head_type] = build_grounding_head(head_type, cfg, *args, **kwargs)
        elif head_type == "TransformerHead":
            heads[head_type] = build_transformer_head(head_type, cfg, *args, **kwargs)
        elif head_type == "MLPHead":
            heads[head_type] = build_mlp_head(head_type, cfg, *args, **kwargs)

    if cfg.MODEL.MMSS_HEAD.TIE_VL_PROJECTION_WEIGHTS:
        weight = heads[cfg.MODEL.MMSS_HEAD.DEFAULT_HEAD].v2l_projection.weight
        bias = heads[cfg.MODEL.MMSS_HEAD.DEFAULT_HEAD].v2l_projection.bias
        for head_type in cfg.MODEL.MMSS_HEAD.TYPES:
            if head_type == cfg.MODEL.MMSS_HEAD.DEFAULT_HEAD:
                continue
            if not hasattr(heads[head_type], "v2l_projection"):
                continue
            assert weight.shape[0] == heads[head_type].v2l_projection.weight.shape[0]
            assert weight.shape[1] == heads[head_type].v2l_projection.weight.shape[1]
            heads[head_type].v2l_projection.weight = weight
            heads[head_type].v2l_projection.bias = bias
    return nn.ModuleDict(heads)
