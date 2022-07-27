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

from ovr.modeling.logged_module import (
    LoggedModule,
    binary_cross_entropy_with_logits,
    normalize_vec,
)

__all__ = [
    "EmbeddingGroundingFastRCNNOutputLayers",
    "GroundingModule",
]

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


class GroundingModule(LoggedModule):
    def __init__(
        self,
        emb_dim,
        num_classes,
        max_tokens,
        local_metric: str = "dot",
        global_metric: str = "aligned_local",
        alignment: str = "softmax",
        temperature: float = 1.0,
        normalize_emb: bool = False,
        background_class: bool = True,
        return_similarity: bool = False,
    ):
        super(GroundingModule, self).__init__()
        self.emb_dim = emb_dim
        self.num_classes = num_classes
        self.max_tokens = max_tokens
        self.local_metric = local_metric
        self.global_metric = global_metric
        self.alignment = alignment
        self.temperature = temperature
        self.normalize_emb = normalize_emb
        self.background_class = background_class
        self.return_similarity = return_similarity
        self.class_emb = torch.zeros(self.num_classes, self.emb_dim)
        self.num_tok = torch.ones(self.num_classes).int()
        self.mask_emb = torch.zeros(self.num_classes, max(self.num_tok))
        self.token_score = None  # nn.Linear(self.emb_dim, self.class_emb.shape[0])
        # nn.init.normal_(self.token_score.weight, mean=0, std=0.01)
        # nn.init.constant_(self.token_score.bias, 0)

        if self.local_metric == "cosine":
            assert self.normalize_emb

    def calc_local_similarity(self, image_emb, num_tok):
        # The shape should be:
        # image_emb.shape == [batch_size, self.emb_dim, num_regions=1 if considering box]
        self.log("image_emb", image_emb)

        if self.local_metric == "dot":
            local_similarity = self.token_score(image_emb)
            local_distance = -local_similarity
        elif self.local_metric == "cosine":
            local_similarity = self.token_score(image_emb)
            # i_norm = (image_emb ** 2).sum(dim=1, keepdim=True).sqrt()
            # self.log('i_norm', i_norm)
            # local_similarity = local_similarity / i_norm
            local_similarity = torch.where(
                torch.isnan(local_similarity),
                torch.zeros_like(local_similarity),
                local_similarity,
            )
            local_distance = 1 - local_similarity
        else:
            raise NotImplementedError

        local_similarity = local_similarity / self.temperature
        local_distance = local_distance / self.temperature
        self.log("local_similarity", local_similarity)
        self.log("local_distance", local_distance)

        # split per class tokens
        split_sizes = num_tok.detach()
        split_sizes[split_sizes == 0] = 1
        split_sizes = split_sizes.tolist()
        local_similarity = torch.split(local_similarity, split_sizes, dim=1)
        local_distance = torch.split(local_distance, split_sizes, dim=1)

        return local_similarity, local_distance

    def calc_global_distance(
        self, image_emb, num_tok, mask_emb, all_loc_sim=None, all_loc_dis=None
    ):
        if all_loc_sim is None or all_loc_dis is None:
            batch_size, _ = image_emb.shape
            local_similarity, local_distance = self.calc_local_similarity(image_emb)
            device = image_emb.device
            all_loc_sim = torch.zeros(
                batch_size, len(num_tok), max(num_tok), device=device
            )
            all_loc_dis = torch.zeros(
                batch_size, len(num_tok), max(num_tok), device=device
            )
            for clss_idx, (loc_sim, loc_dis) in enumerate(
                zip(local_similarity, local_distance)
            ):
                all_loc_sim[:, clss_idx, : num_tok[clss_idx]] = loc_sim
                all_loc_dis[:, clss_idx, : num_tok[clss_idx]] = loc_dis

        self.log("all_loc_sim", all_loc_sim)
        self.log("all_loc_dis", all_loc_dis)
        all_loc_sim = torch.where(
            (mask_emb) > 0, all_loc_sim, all_loc_sim.min().detach() - 100.0
        )

        if self.alignment == "softmax":
            if len(all_loc_sim.shape) == 3:
                tok_attention = F.softmax(all_loc_sim, dim=2)
            elif len(all_loc_sim.shape) == 2:
                tok_attention = F.softmax(all_loc_sim, dim=1)
        elif self.alignment == "hardmax":
            if len(all_loc_sim.shape) == 3:
                idx = torch.argmax(all_loc_sim, dim=2)
            elif len(all_loc_sim.shape) == 2:
                idx = torch.argmax(all_loc_sim, dim=1)
            tok_attention = F.one_hot(idx, max(num_tok)).to(torch.float32)
        self.log("tok_attention", tok_attention)

        if self.global_metric == "aligned_local":
            if len(mask_emb.shape) == 2:
                tok_attention = tok_attention * mask_emb[None, :, :]
                global_dist = (tok_attention * all_loc_dis).sum(dim=2)
            elif len(mask_emb.shape) == 3:
                tok_attention = tok_attention * mask_emb
                global_dist = (tok_attention * all_loc_dis).sum(dim=1)
        else:
            raise NotImplementedError
        self.log("global_dist", global_dist)
        global_dist = torch.where(
            num_tok > 0, global_dist, global_dist.max().detach() + 100.0
        )

        return global_dist, tok_attention

    def forward(self, image_emb):
        batch_size, _ = image_emb.shape
        local_similarity, local_distance = self.calc_local_similarity(
            image_emb, self.num_tok
        )
        device = image_emb.device
        all_loc_sim = torch.zeros(
            batch_size, len(self.num_tok), max(self.num_tok), device=device
        )
        all_loc_dis = torch.zeros(
            batch_size, len(self.num_tok), max(self.num_tok), device=device
        )
        for clss_idx, (loc_sim, loc_dis) in enumerate(
            zip(local_similarity, local_distance)
        ):
            all_loc_sim[:, clss_idx, : self.num_tok[clss_idx]] = loc_sim
            all_loc_dis[:, clss_idx, : self.num_tok[clss_idx]] = loc_dis

        global_dist, tok_attention = self.calc_global_distance(
            image_emb,
            self.num_tok,
            self.mask_emb,
            all_loc_sim=all_loc_sim,
            all_loc_dis=all_loc_dis,
        )

        if self.return_similarity:
            return -global_dist, tok_attention, (local_similarity, local_distance)

        return -global_dist, tok_attention

    def set_class_embeddings(self, embs, device):
        self.num_classes = len(embs)
        if self.background_class:
            num_classes_bg = self.num_classes + 1
        else:
            num_classes_bg = self.num_classes
        self.num_tok = torch.zeros(num_classes_bg, device=device).int()

        all_emb = []
        for cls_idx, cls_emb in embs.items():
            if torch.is_tensor(cls_emb):
                cls_emb = cls_emb.clone().detach().to(device)
            else:
                cls_emb = torch.tensor(cls_emb, device=device)
            len_emb = cls_emb.shape[0]
            self.num_tok[cls_idx] = len_emb
            all_emb.append(cls_emb)

        mask_emb = torch.zeros(num_classes_bg, max(self.num_tok), device=device)
        for cls_idx, len_emb in enumerate(self.num_tok):
            mask_emb[cls_idx, :len_emb] = 1
        self.mask_emb = mask_emb

        # Add background token
        if self.background_class:
            all_emb.append(torch.zeros(1, self.emb_dim, device=device))

        self.class_emb = torch.cat(all_emb, 0)
        self.token_score = nn.Linear(self.emb_dim, self.class_emb.shape[0])
        self.token_score.to(device)

        if self.normalize_emb:
            assert (
                self.class_emb.shape[1] == self.emb_dim
            ), "The embedding dimension has to match the one saved in the model"
            self.class_emb = normalize_vec(self.class_emb, dim=1)

        self.token_score.weight.data = self.class_emb
        self.token_score.bias.data = torch.zeros_like(self.token_score.bias.data)
        self.token_score.weight.requires_grad = False
        self.token_score.bias.requires_grad = False


class EmbeddingGroundingFastRCNNOutputLayers(FastRCNNOutputLayers):
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
        detach_cls_predictor: bool = False,
        grounding_module: None,
        # local_metric: str = "dot",
        # global_metric: str = "aligned_local",
        # alignment: str = "softmax",
        # temperature: float = 1.0,
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
            self.emb_dim = emb_dim
            self.emb_pred = nn.Linear(num_inputs, self.emb_dim)
            nn.init.normal_(self.emb_pred.weight, mean=0, std=0.01)
            nn.init.constant_(self.emb_pred.bias, 0)
            self.cls_score = grounding_module
            assert cls_agnostic_bbox_reg
            # __forward__() can't be used until these are initialized, AFTER the optimizer is made.
            self.num_classes = None

        # if detach_cls_predictor there is no back prop for the cls predictor,
        # useful when classes of objects are unknown
        self.detach_cls_predictor = detach_cls_predictor
        if self.detach_cls_predictor:
            self.loss_weight.update({"loss_cls": 0.0})

    @classmethod
    def from_config(cls, cfg, input_shape):
        grounding_module = GroundingModule(
            cfg.MODEL.ROI_BOX_HEAD.EMB_DIM,
            cfg.MODEL.ROI_HEADS.NUM_CLASSES,
            cfg.MODEL.ROI_HEADS.MAX_TOKENS,
            local_metric=cfg.MODEL.MMSS_HEAD.GROUNDING.LOCAL_METRIC,
            global_metric=cfg.MODEL.MMSS_HEAD.GROUNDING.GLOBAL_METRIC,
            alignment=cfg.MODEL.MMSS_HEAD.GROUNDING.ALIGNMENT,
            temperature=cfg.MODEL.MMSS_HEAD.GROUNDING.ALIGNMENT_TEMPERATURE,
            normalize_emb=cfg.MODEL.ROI_BOX_HEAD.NORMALIZE_EMB_PRED,
        )
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
            "detach_cls_predictor"  : cfg.MODEL.ROI_HEADS.DETACH_CLASS_PREDICTOR,
            
            "grounding_module"      : grounding_module,
            # "local_metric"          : cfg.MODEL.MMSS_HEAD.GROUNDING.LOCAL_METRIC,
            # "global_metric"         : cfg.MODEL.MMSS_HEAD.GROUNDING.GLOBAL_METRIC,
            # "alignment"             : cfg.MODEL.MMSS_HEAD.GROUNDING.ALIGNMENT,
            # "temperature"           : cfg.MODEL.MMSS_HEAD.GROUNDING.ALIGNMENT_TEMPERATURE,
            # fmt: on
        }

        return args_dict

    def device(self):
        device = self.emb_pred.weight.device
        return device

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
            scores, _ = self.cls_score(x)
        else:
            scores = self.cls_score(x)
        return scores

    def set_class_embeddings(self, embs):
        self.cls_score.set_class_embeddings(embs, self.device())
        self.num_classes = self.cls_score.num_classes
