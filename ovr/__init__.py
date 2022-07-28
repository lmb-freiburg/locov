# Register models
from ovr.modeling.meta_arch.mmss_gcnn import MMSSGridModel
from ovr.modeling.meta_arch.distill_mmss_gcnn import DistillMMSSGridModel
from ovr.modeling.meta_arch.distill_prop_mmss_gcnn import (
    DistillProposalMMSSRCNN,
    DistillOnlyProposalMMSSRCNN,
)
from ovr.modeling.meta_arch.ovr_rcnn import OvrRCNN
from ovr.modeling.roi_heads.roi_emb_heads import EmbeddingRes5ROIHeads
from ovr.modeling.roi_heads.roi_emb_heads import EmbeddingProposalsRes5ROIHeads
