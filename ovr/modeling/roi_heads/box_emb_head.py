# Copyright (c) Facebook, Inc. and its affiliates.
import logging
from typing import Dict, List, Tuple, Union
import torch
from fvcore.nn import giou_loss, smooth_l1_loss
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.layers import ShapeSpec, batched_nms, cat, cross_entropy, nonzero_tuple
from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.modeling.roi_heads.fast_rcnn import (
    fast_rcnn_inference,
    fast_rcnn_inference_single_image,
    FastRCNNOutputLayers,
    _log_classification_stats,
)
from detectron2.structures import Boxes, Instances
from detectron2.utils.events import get_event_storage

from ovr.modeling.roi_heads.box_emb_grounding_head import (
    EmbeddingGroundingFastRCNNOutputLayers,
)
from ovr.modeling.logged_module import (
    binary_cross_entropy_with_logits,
    normalize_vec,
    standardize_vec,
)

__all__ = ["EmbeddingFastRCNNOutputLayers"]

logger = logging.getLogger(__name__)

"""
Shape shorthand in this module:

    N: number of images in the minibatch
    R: number of ROIs, combined over all images, in the minibatch
    Ri: number of ROIs in image i
    K: number of foreground classes. E.g.,there are 80 foreground classes in COCO.

Naming convention:

    deltas: refers to the 4-d (dx, dy, dw, dh) deltas that parameterize the box2box
    transform (see :class:`box_regression.Box2BoxTransform`).

    pred_class_logits: predicted class scores in [-inf, +inf]; use
        softmax(pred_class_logits) to estimate P(class).

    gt_classes: ground-truth classification labels in [0, K], where [0, K) represent
        foreground object classes and K represents the background class.

    pred_proposal_deltas: predicted box2box transform deltas for transforming proposals
        to detection box predictions.

    gt_proposal_deltas: ground-truth box2box transform deltas
"""


class EmbeddingFastRCNNOutputLayers(FastRCNNOutputLayers):
    """
    Two linear layers for predicting Fast R-CNN outputs:

    1. proposal-to-detection box regression deltas
    2. classification scores
    """

    @configurable
    def __init__(
        self,
        input_shape: ShapeSpec,
        *,
        box2box_transform,
        num_classes: int,
        test_score_thresh: float = 0.0,
        test_nms_thresh: float = 0.5,
        test_topk_per_image: int = 100,
        cls_agnostic_bbox_reg: bool = False,
        smooth_l1_beta: float = 0.0,
        box_reg_loss_type: str = "smooth_l1",
        loss_weight: Union[float, Dict[str, float]] = 1.0,
        emb_dim: int = 768,
        embedding_based: bool = True,
        freeze_emb_pred: bool = True,
        normalize_emb: bool = False,
        standardize_emb: bool = False,
        detach_cls_predictor: bool = False,
    ):
        """
        NOTE: this interface is experimental.

        Args:
            input_shape (ShapeSpec): shape of the input feature to this module
            box2box_transform (Box2BoxTransform or Box2BoxTransformRotated):
            num_classes (int): number of foreground classes
            test_score_thresh (float): threshold to filter predictions results.
            test_nms_thresh (float): NMS threshold for prediction results.
            test_topk_per_image (int): number of top predictions to produce per image.
            cls_agnostic_bbox_reg (bool): whether to use class agnostic for bbox regression
            smooth_l1_beta (float): transition point from L1 to L2 loss. Only used if
                `box_reg_loss_type` is "smooth_l1"
            box_reg_loss_type (str): Box regression loss type. One of: "smooth_l1", "giou"
            loss_weight (float|dict): weights to use for losses. Can be single float for weighting
                all losses, or a dict of individual weightings. Valid dict keys are:
                    * "loss_cls": applied to classification loss
                    * "loss_box_reg": applied to box regression loss
        """
        super().__init__(
            input_shape,
            box2box_transform=box2box_transform,
            num_classes=num_classes,
            test_score_thresh=test_score_thresh,
            test_nms_thresh=test_nms_thresh,
            test_topk_per_image=test_topk_per_image,
            cls_agnostic_bbox_reg=cls_agnostic_bbox_reg,
            smooth_l1_beta=smooth_l1_beta,
            box_reg_loss_type=box_reg_loss_type,
            loss_weight=loss_weight,
        )

        if isinstance(input_shape, int):  # some backward compatibility
            input_shape = ShapeSpec(channels=input_shape)
        num_inputs = (
            input_shape.channels * (input_shape.width or 1) * (input_shape.height or 1)
        )
        # prediction layer for num_classes foreground classes and one background class (hence + 1)

        # Add the embedding based layer
        self.embedding_based = embedding_based
        if self.embedding_based:
            self.normalize_emb = normalize_emb
            self.standardize_emb = standardize_emb
            self.emb_dim = emb_dim
            self.emb_pred = nn.Linear(num_inputs, self.emb_dim)
            nn.init.normal_(self.emb_pred.weight, mean=0, std=0.01)
            nn.init.constant_(self.emb_pred.bias, 0)
            assert cls_agnostic_bbox_reg
            # __forward__() can't be used until these are initialized, AFTER the optimizer is made.
            self.num_classes = None
            self.cls_score = None
            if freeze_emb_pred:
                self.emb_pred.weight.requires_grad = False
                self.emb_pred.bias.requires_grad = False

        # if detach_cls_predictor there is no back prop for the cls predictor,
        # useful when classes of objects are unknown
        self.detach_cls_predictor = detach_cls_predictor
        if self.detach_cls_predictor:
            self.loss_weight.update({"loss_cls": 0.0})

    @classmethod
    def from_config(cls, cfg, input_shape):
        args_dict = {
            "input_shape": input_shape,
            "box2box_transform": Box2BoxTransform(
                weights=cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_WEIGHTS
            ),
            # fmt: off
            "num_classes"           : cfg.MODEL.ROI_HEADS.NUM_CLASSES,
            "cls_agnostic_bbox_reg" : cfg.MODEL.ROI_BOX_HEAD.CLS_AGNOSTIC_BBOX_REG,
            "smooth_l1_beta"        : cfg.MODEL.ROI_BOX_HEAD.SMOOTH_L1_BETA,
            "test_score_thresh"     : cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST,
            "test_nms_thresh"       : cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST,
            "test_topk_per_image"   : cfg.TEST.DETECTIONS_PER_IMAGE,
            "box_reg_loss_type"     : cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_LOSS_TYPE,
            "loss_weight"           : {"loss_box_reg": cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_LOSS_WEIGHT},

            "emb_dim"               : cfg.MODEL.ROI_BOX_HEAD.EMB_DIM,
            "embedding_based"       : cfg.MODEL.ROI_BOX_HEAD.EMBEDDING_BASED,
            "freeze_emb_pred"       : cfg.MODEL.ROI_BOX_HEAD.FREEZE_EMB_PRED,
            "normalize_emb"         : cfg.MODEL.ROI_BOX_HEAD.NORMALIZE_EMB_PRED,     
            "standardize_emb"       : cfg.MODEL.ROI_BOX_HEAD.STANDARDIZE_EMB_PRED,            
            "detach_cls_predictor"  : cfg.MODEL.ROI_HEADS.DETACH_CLASS_PREDICTOR,
            # fmt: on
        }

        return args_dict

    def forward(self, x):
        """
        Args:
            x: per-region features of shape (N, ...) for N bounding boxes to predict.

        Returns:
            (Tensor, Tensor):
            First tensor: shape (N,K+1), scores for each of the N box. Each row contains the
            scores for K object categories and 1 background class.

            Second tensor: bounding box regression deltas for each box. Shape is shape (N,Kx4),
            or (N,4) for class-agnostic regression.
        """
        if x.dim() > 2:
            x = torch.flatten(x, start_dim=1)
        # x = self.avgpool(x)
        # x = x.view(x.size(0), -1)
        proposal_deltas = self.bbox_pred(x)
        if self.detach_cls_predictor:
            with torch.no_grad():
                scores = self.forward_cls_prediction(x.detach())
        else:
            scores = self.forward_cls_prediction(x)
        return scores, proposal_deltas

    def forward_cls_prediction(self, x):
        if self.embedding_based:
            x = self.emb_pred(x)
            if self.normalize_emb:
                x = normalize_vec(x, dim=1)
            if self.standardize_emb:
                x = standardize_vec(x, dim=1)
        scores = self.cls_score(x)
        return scores

    def set_class_embeddings(self, embs):
        device = self.emb_pred.weight.device
        self.num_classes = embs.shape[0] - 1  # Includes background
        self.cls_score = nn.Linear(self.emb_dim, self.num_classes + 1)
        self.cls_score.to(device)
        if torch.is_tensor(embs):
            embs = embs.clone().detach().to(device)
        else:
            embs = torch.tensor(embs, device=device)
        if self.normalize_emb:
            assert (
                embs.shape[1] == self.emb_dim
            ), "The embedding dimension has to match the one saved in the model"
            embs = normalize_vec(embs, dim=1)
        if self.standardize_emb:
            assert (
                embs.shape[1] == self.emb_dim
            ), "The embedding dimension has to match the one saved in the model"
            embs = standardize_vec(embs, dim=1)
        self.cls_score.weight.data = embs
        self.cls_score.bias.data = torch.zeros_like(self.cls_score.bias.data)
        self.cls_score.weight.requires_grad = False
        self.cls_score.bias.requires_grad = False


def build_box_predictor(cfg, input_shape):
    """
    Build a box head defined by `cfg.MODEL.ROI_BOX_HEAD.NAME`.
    """
    name = cfg.MODEL.ROI_BOX_HEAD.NAME
    BOX_EMBEDDING_PREDICTORS = {
        "FastRCNNOutputLayers": FastRCNNOutputLayers,
        "EmbeddingFastRCNNOutputLayers": EmbeddingFastRCNNOutputLayers,
        "EmbeddingGroundingFastRCNNOutputLayers": EmbeddingGroundingFastRCNNOutputLayers,
    }
    return BOX_EMBEDDING_PREDICTORS[name](cfg, input_shape)
