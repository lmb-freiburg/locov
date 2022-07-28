from collections import OrderedDict
import torch
from torch import nn
import ovr.modeling.language.transf_models as transf

from detectron2.utils.registry import Registry

LANGUAGE_BACKBONES_REGISTRY = Registry("LANGUAGE_BACKBONES")
LANGUAGE_BACKBONES_REGISTRY.__doc__ = """
Registry for language backbones.

The registered object will be called with `obj(cfg)`
and expected to return a `nn.Module` object.
"""

__all__ = [
    "build_backbone",
    "build_bert_backbone",
    "build_bertemb_backbone",
]


@LANGUAGE_BACKBONES_REGISTRY.register()
def build_bert_backbone(cfg):
    body = transf.BERT(cfg)
    model = nn.Sequential(OrderedDict([("body", body)]))
    model.out_channels = body.out_channels
    return model


@LANGUAGE_BACKBONES_REGISTRY.register()
def build_bertemb_backbone(cfg):
    body = transf.BertEmbedding(cfg)
    model = nn.Sequential(OrderedDict([("body", body)]))
    model.out_channels = body.out_channels
    return model


def build_backbone(cfg):
    assert (
        cfg.MODEL.LANGUAGE_BACKBONE.TYPE in LANGUAGE_BACKBONES_REGISTRY
    ), "cfg.LANGUAGE_MODEL.TYPE: {} is not registered in registry".format(
        cfg.MODEL.LANGUAGE_BACKBONE.TYPE
    )
    lang_arch = cfg.MODEL.LANGUAGE_BACKBONE.TYPE
    model = LANGUAGE_BACKBONES_REGISTRY.get(lang_arch)(cfg)
    model.to(torch.device(cfg.MODEL.DEVICE))
    return model
