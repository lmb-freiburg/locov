"""
Implements the Multimedia Self-Supervised Grid-based (proposal-free) CNN framework
Edited from https://github.com/alirezazareian/ovr-cnn/maskrcnn_benchmark/modeling/detector/mmss_gcnn.py
"""
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
from torch import nn

from detectron2.modeling.backbone import build_backbone
from detectron2.modeling import META_ARCH_REGISTRY
from detectron2.structures import ImageList, Instances

from ovr.modeling.language.backbone import build_backbone as build_language_backbone
from ovr.modeling.mmss_heads.mmss_heads import build_mmss_heads
from ovr.modeling.logged_module import LoggedModule

__all__ = ["MMSSGridModel"]


@META_ARCH_REGISTRY.register()
class MMSSGridModel(LoggedModule):  # (nn.Module):
    """
    Main class for Generalized R-CNN. Currently supports boxes and masks.
    It consists of three main parts:
    - backbone
    - rpn
    - heads: takes the features + the proposals from the RPN and computes
        detections / masks from it.
    """

    def __init__(self, cfg):
        super(MMSSGridModel, self).__init__()

        self.backbone = build_backbone(cfg)
        self.language_backbone = build_language_backbone(cfg)
        self.vis_in_features = cfg.MODEL.MMSS_HEAD.IN_FEATURES
        self.mmss_heads = build_mmss_heads(
            cfg,
            v_dim=self.backbone.output_shape()[self.vis_in_features].channels,
            l_dim=self.language_backbone.out_channels,
            loc_dim=2,
            backbone=self.language_backbone.body,
        )
        self.mvm = cfg.MODEL.MMSS_HEAD.TRANSFORMER.MASKED_VISUAL_MODELING
        self.spatial_dropout = cfg.MODEL.MMSS_HEAD.SPATIAL_DROPOUT

        self.register_buffer(
            "pixel_mean", torch.Tensor(cfg.MODEL.PIXEL_MEAN).view(-1, 1, 1)
        )
        self.register_buffer(
            "pixel_std", torch.Tensor(cfg.MODEL.PIXEL_STD).view(-1, 1, 1)
        )
        assert (
            self.pixel_mean.shape == self.pixel_std.shape
        ), f"{self.pixel_mean} and {self.pixel_std} have different shapes!"

    @property
    def device(self):
        return self.pixel_mean.device

    def preprocess_image(self, batched_inputs: Tuple[Dict[str, torch.Tensor]]):
        """
        Normalize, pad and batch the input images.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        return images

    def preprocess_text(self, batched_inputs: Tuple[Dict[str, torch.Tensor]]):
        text_list = [x["caption"] for x in batched_inputs]
        return text_list

    def forward(self, batched_inputs: Tuple[Dict[str, torch.Tensor]]):
        """
        Arguments:
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances (optional): groundtruth :class:`Instances`
                * proposals (optional): :class:`Instances`, precomputed proposals.

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.

        Returns:
            result tuple: (dict[Tensor], dict[Tensor]): losses and other information.

        """
        images = self.preprocess_image(batched_inputs)
        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None

        visual_grid_features = self.backbone(images.tensor)[self.vis_in_features]
        self.log("images.tensor", images.tensor)
        self.log("visual_grid_features", visual_grid_features)

        _, _, image_h, image_w = images.tensor.shape
        batch_size, dim, grid_h, grid_w = visual_grid_features.shape
        max_num_regions = grid_h * grid_w

        flattened_features = visual_grid_features.reshape(
            [batch_size, dim, max_num_regions]
        ).permute(0, 2, 1)

        image_sizes = np.asarray(images.image_sizes, dtype=np.float32)
        grid_sizes = np.zeros(image_sizes.shape, dtype=np.int32)
        grid_sizes[:, 0] = np.ceil(image_sizes[:, 0] * grid_h / image_h)
        grid_sizes[:, 1] = np.ceil(image_sizes[:, 1] * grid_w / image_w)
        grid_mask = np.zeros([batch_size, grid_h, grid_w], dtype=np.uint8)
        for i in range(batch_size):
            grid_mask[i, : grid_sizes[i, 0], : grid_sizes[i, 1]] = 1
        flattened_mask = grid_mask.reshape([batch_size, max_num_regions])

        loc_x = np.zeros([batch_size, grid_h, grid_w], dtype=np.float32)
        loc_y = np.zeros([batch_size, grid_h, grid_w], dtype=np.float32)
        for i in range(batch_size):
            y = (np.arange(grid_sizes[i, 0], dtype=np.float32) + 0.5) / grid_sizes[i, 0]
            x = (np.arange(grid_sizes[i, 1], dtype=np.float32) + 0.5) / grid_sizes[i, 1]
            loc_x[i, : grid_sizes[i, 0], : grid_sizes[i, 1]] = x[None, :]
            loc_y[i, : grid_sizes[i, 0], : grid_sizes[i, 1]] = y[:, None]
        flattened_loc = np.stack([loc_x, loc_y], axis=-1).reshape(
            [batch_size, max_num_regions, 2]
        )
        flattened_loc = torch.tensor(flattened_loc).cuda()

        if self.spatial_dropout > 0 and self.training:
            subsampled_features = []
            subsampled_loc = []
            new_mask = np.zeros([batch_size, self.spatial_dropout], dtype=np.uint8)
            for i in range(batch_size):
                idx = np.where(flattened_mask[i])[0]
                np.random.shuffle(idx)
                n = min(self.spatial_dropout, idx.shape[0])
                idx = idx[:n]
                subsampled_features.append(flattened_features[i, idx])
                subsampled_loc.append(flattened_loc[i, idx])
                new_mask[i, :n] = 1
            flattened_features = torch.nn.utils.rnn.pad_sequence(
                subsampled_features, batch_first=True
            )
            flattened_loc = torch.nn.utils.rnn.pad_sequence(
                subsampled_loc, batch_first=True
            )
            flattened_mask = new_mask

        input_image = {
            "region_features": flattened_features,
            "region_mask": torch.tensor(flattened_mask).cuda(),
            "region_loc": flattened_loc,
            "mvm_mask": torch.zeros(batch_size, max_num_regions).cuda(),
            "target_region_features": flattened_features,
        }
        # print('images.tensor.shape', images.tensor.shape)
        # print('visual_grid_features.shape', visual_grid_features.shape)
        # print('flattened_features', flattened_features.shape)
        if self.mvm:
            raise NotImplementedError

        targets = self.preprocess_text(batched_inputs)
        input_caption = self.language_backbone(targets)
        # print('images.tensor', images.tensor.shape)
        # print('max_num_regions', max_num_regions)
        # for k,v in input_image.items():
        #     print(k, v.shape)
        # for k,v in input_caption.items():
        #     print(k, v.shape)

        mmss_outputs = {}
        mmss_losses = {}
        for head in self.mmss_heads:
            o, l = self.mmss_heads[head](input_image, input_caption)
            mmss_outputs.update(o)
            mmss_losses.update(l)

        for v in l.values():
            if torch.isnan(v):
                print("MMSSGridModel", self.log_info)
                for head in self.mmss_heads:
                    print(head, self.mmss_heads[head].log_info)
                print("image_sizes", image_sizes, "grid_sizes", grid_sizes)
                raise ValueError()

        return mmss_outputs, mmss_losses
