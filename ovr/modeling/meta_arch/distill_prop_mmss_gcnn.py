import logging
import numpy as np
from typing import Dict, List, Optional, Tuple
import torch
from torch import nn

from detectron2.config import configurable
from detectron2.modeling.meta_arch.rcnn import GeneralizedRCNN
from detectron2.modeling.backbone import Backbone, build_backbone
from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.modeling.proposal_generator import build_proposal_generator
from detectron2.modeling.roi_heads import build_roi_heads
from detectron2.modeling import META_ARCH_REGISTRY
from detectron2.structures import ImageList, Instances
from detectron2.data.detection_utils import convert_image_to_rgb

from ovr.modeling.language.backbone import build_backbone as build_language_backbone
from ovr.modeling.logged_module import LoggedModule
from ovr.modeling.mmss_heads.mmss_heads import build_mmss_heads
from ovr.modeling.meta_arch.distill_mmss_gcnn import (
    MultiDistillLoss,
    MultiDistillLossJS,
    MultiDistillLossL2,
)

__all__ = ["DistillProposalMMSSRCNN", "DistillOnlyProposalMMSSRCNN"]


@META_ARCH_REGISTRY.register()
class DistillProposalMMSSRCNN(LoggedModule):
    """
    Generalized R-CNN. Any models that contains the following three components:
    1. Per-image feature extraction (aka backbone)
    2. Region proposal generation
    3. Per-region feature extraction and prediction
    4. Language model transformer
    5. Multimodal transformer
    6. Distillation loss between grounding and multimodal transformer
    """

    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        proposal_generator: nn.Module,
        roi_heads: nn.Module,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        language_backbone: nn.Module,
        mmss_heads: nn.Module,
        distill_loss: Optional[nn.Module] = None,
        input_format: Optional[str] = None,
        vis_period: int = 0,
        vis_in_features: str = "res5",
        mvm: Optional[bool] = False,
        spatial_dropout: int = 100,
    ):
        """
        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            proposal_generator: a module that generates proposals using backbone features
            roi_heads: a ROI head that performs per-region computation
            pixel_mean, pixel_std: list or tuple with #channels element, representing
                the per-channel mean and std to be used to normalize the input image
            input_format: describe the meaning of channels of input. Needed by visualization
            vis_period: the period to run visualization. Set to 0 to disable.
        """
        super().__init__()
        # Normal Detection setup
        self.backbone = backbone
        self.proposal_generator = proposal_generator
        self.roi_heads = roi_heads

        self.input_format = input_format
        self.vis_period = vis_period
        if vis_period > 0:
            assert (
                input_format is not None
            ), "input_format is required for visualization!"

        self.register_buffer(
            "pixel_mean", torch.tensor(pixel_mean).view(-1, 1, 1), False
        )
        self.register_buffer("pixel_std", torch.tensor(pixel_std).view(-1, 1, 1), False)
        assert (
            self.pixel_mean.shape == self.pixel_std.shape
        ), f"{self.pixel_mean} and {self.pixel_std} have different shapes!"

        # MMSS setup
        self.language_backbone = language_backbone
        self.vis_in_features = vis_in_features
        self.mmss_heads = mmss_heads
        self.mvm = mvm
        self.spatial_dropout = spatial_dropout
        self.distill_loss = distill_loss

        self.init_model_features()

    def init_model_features(self):
        return

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        roi_heads = build_roi_heads(cfg, backbone.output_shape())
        language_backbone = build_language_backbone(cfg)
        vis_in_features = cfg.MODEL.MMSS_HEAD.IN_FEATURES
        mmss_heads = build_mmss_heads(
            cfg,
            v_dim=roi_heads.output_shape,
            l_dim=language_backbone.out_channels,
            loc_dim=2,
            backbone=language_backbone.body,
        )
        # Tie embedding prediction layer of detector with v2l_projection layer
        if cfg.MODEL.LOAD_EMB_PRED_FROM_MMSS_HEAD:
            weight = mmss_heads[cfg.MODEL.MMSS_HEAD.DEFAULT_HEAD].v2l_projection.weight
            bias = mmss_heads[cfg.MODEL.MMSS_HEAD.DEFAULT_HEAD].v2l_projection.bias

            assert hasattr(roi_heads.box_predictor, "emb_pred")
            assert weight.shape[0] == roi_heads.box_predictor.emb_pred.weight.shape[0]
            assert weight.shape[1] == roi_heads.box_predictor.emb_pred.weight.shape[1]
            roi_heads.box_predictor.emb_pred.weight = weight
            roi_heads.box_predictor.emb_pred.bias = bias

        if cfg.MODEL.MMSS_HEAD.DISTILLATION_LOSS:
            if cfg.MODEL.MMSS_HEAD.DISTILLATION_LOSS_TYPE == "KD":
                distill_loss = MultiDistillLoss(
                    cfg.MODEL.MMSS_HEAD.DISTILLATION_TEMPERATURE,
                    cfg.MODEL.MMSS_HEAD.DISTILLATION_LOSS_WEIGHT,
                    cfg.MODEL.MMSS_HEAD.DISTILLATION_DETACH_TEACHER,
                    cfg.MODEL.MMSS_HEAD.DISTILLATION_TEACHER_TRANSFORMER,
                )
            elif cfg.MODEL.MMSS_HEAD.DISTILLATION_LOSS_TYPE == "JS":
                distill_loss = MultiDistillLossJS(
                    cfg.MODEL.MMSS_HEAD.DISTILLATION_TEMPERATURE,
                    cfg.MODEL.MMSS_HEAD.DISTILLATION_LOSS_WEIGHT,
                    cfg.MODEL.MMSS_HEAD.DISTILLATION_DETACH_TEACHER,
                    cfg.MODEL.MMSS_HEAD.DISTILLATION_TEACHER_TRANSFORMER,
                )
            elif cfg.MODEL.MMSS_HEAD.DISTILLATION_LOSS_TYPE == "MSE":
                distill_loss = MultiDistillLossL2(
                    cfg.MODEL.MMSS_HEAD.DISTILLATION_TEMPERATURE,
                    cfg.MODEL.MMSS_HEAD.DISTILLATION_LOSS_WEIGHT,
                    cfg.MODEL.MMSS_HEAD.DISTILLATION_DETACH_TEACHER,
                    cfg.MODEL.MMSS_HEAD.DISTILLATION_TEACHER_TRANSFORMER,
                )
        else:
            distill_loss = None

        return {
            "backbone": backbone,
            "proposal_generator": build_proposal_generator(
                cfg, backbone.output_shape()
            ),
            "roi_heads": roi_heads,
            "input_format": cfg.INPUT.FORMAT,
            "vis_period": cfg.VIS_PERIOD,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            # MMSS setup
            "language_backbone": language_backbone,
            "vis_in_features": vis_in_features,
            "mmss_heads": mmss_heads,
            "mvm": cfg.MODEL.MMSS_HEAD.TRANSFORMER.MASKED_VISUAL_MODELING,
            "spatial_dropout": cfg.MODEL.MMSS_HEAD.SPATIAL_DROPOUT,
            "distill_loss": distill_loss,
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def visualize_training(self, batched_inputs, proposals):
        """
        A function used to visualize images and proposals. It shows ground truth
        bounding boxes on the original image and up to 20 top-scoring predicted
        object proposals on the original image. Users can implement different
        visualization functions for different models.

        Args:
            batched_inputs (list): a list that contains input to the model.
            proposals (list): a list that contains predicted proposals. Both
                batched_inputs and proposals should have the same length.
        """
        from detectron2.utils.visualizer import Visualizer

        storage = get_event_storage()
        max_vis_prop = 20

        for input, prop in zip(batched_inputs, proposals):
            img = input["image"]
            img = convert_image_to_rgb(img.permute(1, 2, 0), self.input_format)
            v_gt = Visualizer(img, None)
            v_gt = v_gt.overlay_instances(boxes=input["instances"].gt_boxes)
            anno_img = v_gt.get_image()
            box_size = min(len(prop.proposal_boxes), max_vis_prop)
            v_pred = Visualizer(img, None)
            v_pred = v_pred.overlay_instances(
                boxes=prop.proposal_boxes[0:box_size].tensor.cpu().numpy()
            )
            prop_img = v_pred.get_image()
            vis_img = np.concatenate((anno_img, prop_img), axis=1)
            vis_img = vis_img.transpose(2, 0, 1)
            vis_name = "Left: GT bounding boxes;  Right: Predicted proposals"
            storage.put_image(vis_name, vis_img)
            break  # only visualize one image in a batch

    def forward(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        """
        Arguments:
            images (list[Tensor] or ImageList): images to be processed
            targets (list[BoxList]): ground-truth boxes present in the image (optional)
                [or (list[ndarray]) with image-level labels in weakly supervised settings]

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        # Inference
        if not self.training:
            return self.inference(batched_inputs)

        # Language backbone
        targets = self.preprocess_text(batched_inputs)
        input_caption = self.language_backbone(targets)
        # import ipdb; ipdb.set_trace()

        # Visual backbone
        images = self.preprocess_image(batched_inputs)
        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None

        features = self.backbone(images.tensor)

        # Run the RPN
        if self.proposal_generator is not None:
            proposals, proposal_losses = self.proposal_generator(
                images, features, gt_instances
            )
        else:
            assert "proposals" in batched_inputs[0]
            proposals = [x["proposals"].to(self.device) for x in batched_inputs]
            proposal_losses = {}

        # Normal detector
        (
            visual_grid_features,
            box_features,
            box_proposals,
            detector_losses,
        ) = self.roi_heads(images, features, proposals, gt_instances)

        if self.vis_period > 0:
            storage = get_event_storage()
            if storage.iter % self.vis_period == 0:
                self.visualize_training(batched_inputs, box_proposals)

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)

        # MMSS
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
            "region_mask": torch.tensor(flattened_mask).to(self.device),
            "region_loc": flattened_loc,
            "mvm_mask": torch.zeros(batch_size, max_num_regions).to(self.device),
            "target_region_features": flattened_features,
        }

        if self.mvm:
            raise NotImplementedError

        # Process heads on whole image grid
        mmss_outputs = {}
        mmss_losses = {}
        mmss_distributions = {}
        for head in self.mmss_heads:
            if self.distill_loss is not None:
                o, l, d = self.mmss_heads[head](input_image, input_caption)
                mmss_outputs.update(o)
                mmss_losses.update(l)
                mmss_distributions.update(d)
            else:
                o, l = self.mmss_heads[head](input_image, input_caption)
                mmss_outputs.update(o)
                mmss_losses.update(l)

        # Grounding BoxMMSS
        num_boxes = [len(box) for box in box_proposals]
        num_boxes = min(num_boxes)
        if self.spatial_dropout > 0 and self.training:
            num_boxes = min(num_boxes, self.spatial_dropout)

        subsampled_box_features = []
        subsampled_box_proposals = []
        for i_box, box in enumerate(box_proposals):
            idx = np.arange(len(box))
            np.random.shuffle(idx)
            idx = idx[:num_boxes]
            subsampled_box_proposals.append(box[idx])
            subsampled_box_features.append(box_features[i_box][idx])

        if self.vis_period > 0:
            storage = get_event_storage()
            if storage.iter % self.vis_period == 0:
                self.visualize_training(batched_inputs, subsampled_box_proposals)

        # The box centers in a Nx2 array of (x, y).
        subsampled_box_centers = [
            prop.get("proposal_boxes").get_centers()
            for prop in subsampled_box_proposals
        ]
        # Normalize centers to [0,1]
        subsampled_box_loc = [
            torch.stack(
                [
                    center[:, 0] / prop._image_size[1],
                    center[:, 1] / prop._image_size[0],
                ],
                dim=-1,
            )
            for center, prop in zip(subsampled_box_centers, subsampled_box_proposals)
        ]

        flattened_box_features = torch.nn.utils.rnn.pad_sequence(
            subsampled_box_features, batch_first=True
        )
        flattened_box_loc = torch.nn.utils.rnn.pad_sequence(
            subsampled_box_loc, batch_first=True
        )
        flattened_box_mask = np.ones([batch_size, num_boxes], dtype=np.uint8)

        input_boxes = {
            "region_features": flattened_box_features,
            "region_mask": torch.tensor(flattened_box_mask).to(self.device),
            "region_loc": flattened_box_loc,
            "mvm_mask": torch.zeros(batch_size, num_boxes).to(self.device),
            "target_region_features": flattened_box_features,
        }
        if self.mvm:
            raise NotImplementedError

        for head in self.mmss_heads:
            if self.distill_loss is not None:
                box_o, box_l, box_d = self.mmss_heads[head](input_boxes, input_caption)
                box_o = {"Box " + k: v for k, v in box_o.items()}
                box_l = {"Box " + k: v for k, v in box_l.items()}
                box_d = {"box_" + k: v for k, v in box_d.items()}
                mmss_outputs.update(box_o)
                mmss_losses.update(box_l)
                mmss_distributions.update(box_d)
            else:
                box_o, box_l = self.mmss_heads[head](input_boxes, input_caption)
                box_o = {"Box " + k: v for k, v in box_o.items()}
                box_l = {"Box " + k: v for k, v in box_l.items()}
                mmss_outputs.update(box_o)
                mmss_losses.update(box_l)

        del input_boxes
        del input_caption
        del input_image

        # Perform distillation loss
        if self.distill_loss is not None:
            trans_pw_cost = mmss_distributions["trans"]
            w2r_pw_cost = mmss_distributions["w2r"]
            r2w_pw_cost = mmss_distributions["r2w"]
            kd_loss = self.distill_loss(trans_pw_cost, w2r_pw_cost, r2w_pw_cost)
            mmss_losses["kd_loss"] = kd_loss

            box_trans_pw_cost = mmss_distributions["box_trans"]
            box_w2r_pw_cost = mmss_distributions["box_w2r"]
            box_r2w_pw_cost = mmss_distributions["box_r2w"]
            box_kd_loss = self.distill_loss(
                box_trans_pw_cost, box_w2r_pw_cost, box_r2w_pw_cost
            )
            mmss_losses["box_kd_loss"] = box_kd_loss

            mixbox_kd_loss = self.distill_loss(
                trans_pw_cost, box_w2r_pw_cost, box_r2w_pw_cost
            )
            mmss_losses["mixbox_kd_loss"] = mixbox_kd_loss

        for v in mmss_losses.values():
            if torch.isnan(v):
                for head in self.mmss_heads:
                    print(head, self.mmss_heads[head].log_info)
                print(image_sizes, grid_sizes)
                raise ValueError()

        losses.update(mmss_losses)

        # if False:
        #     ids = [sample['image_id'] for sample in batched_inputs]
        #     captions = [sample['caption'] for sample in batched_inputs]
        #     trans = torch.diag(mmss_distributions['trans']).detach()
        #     w2r = torch.diag(mmss_distributions['w2r']).detach()
        #     r2w = torch.diag(mmss_distributions['r2w']).detach()

        #     box_trans = torch.diag(mmss_distributions['box_trans']).detach()
        #     box_w2r = torch.diag(mmss_distributions['box_w2r']).detach()
        #     box_r2w = torch.diag(mmss_distributions['box_r2w']).detach()

        #     #  image_id; caption; trans; w2r; r2w; box_trans; box_w2r; box_r2w;
        #     csv_file = "/home/bravoma/repos/new_wsog_ovr/no_distill_dis.csv"
        #     data_array = [[ids[samp], captions[samp], trans[samp].item(), w2r[samp].item(), \
        #         r2w[samp].item(), box_trans[samp].item(), box_w2r[samp].item(), box_r2w[samp].item()] \
        #             for samp in range(len(ids))]
        #     print(data_array)
        #     import ipdb; ipdb.set_trace()
        # with open(csv_file, 'a') as fh:
        #     for sample_data in data_array:
        #         for data_item in sample_data:
        #             fh.write(str(data_item)+ ";")
        #         fh.write('\n')

        return mmss_outputs, losses

    def inference(
        self,
        batched_inputs: List[Dict[str, torch.Tensor]],
        detected_instances: Optional[List[Instances]] = None,
        do_postprocess: bool = True,
    ):
        """
        Run inference on the given inputs.

        Args:
            batched_inputs (list[dict]): same as in :meth:`forward`
            detected_instances (None or list[Instances]): if not None, it
                contains an `Instances` object per image. The `Instances`
                object contains "pred_boxes" and "pred_classes" which are
                known boxes in the image.
                The inference will then skip the detection of bounding boxes,
                and only predict other per-ROI outputs.
            do_postprocess (bool): whether to apply post-processing on the outputs.

        Returns:
            When do_postprocess=True, same as in :meth:`forward`.
            Otherwise, a list[Instances] containing raw network outputs.
        """
        assert not self.training

        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)

        if detected_instances is None:
            if self.proposal_generator is not None:
                proposals, _ = self.proposal_generator(images, features, None)
            else:
                assert "proposals" in batched_inputs[0]
                proposals = [x["proposals"].to(self.device) for x in batched_inputs]

            results, _ = self.roi_heads(images, features, proposals, None)
        else:
            detected_instances = [x.to(self.device) for x in detected_instances]
            results = self.roi_heads.forward_with_given_boxes(
                features, detected_instances
            )

        if do_postprocess:
            assert (
                not torch.jit.is_scripting()
            ), "Scripting is not supported for postprocess."
            return self._postprocess(results, batched_inputs, images.image_sizes)
        else:
            return results

    def preprocess_text(self, batched_inputs: Tuple[Dict[str, torch.Tensor]]):
        text_list = [x["caption"] for x in batched_inputs]
        return text_list

    def preprocess_image(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        """
        Normalize, pad and batch the input images.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        return images

    @staticmethod
    def _postprocess(
        instances, batched_inputs: List[Dict[str, torch.Tensor]], image_sizes
    ):
        """
        Rescale the output instances to the target size.
        """
        # note: private function; subject to changes
        processed_results = []
        for results_per_image, input_per_image, image_size in zip(
            instances, batched_inputs, image_sizes
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            r = detector_postprocess(results_per_image, height, width)
            processed_results.append({"instances": r})
        return processed_results


@META_ARCH_REGISTRY.register()
class DistillOnlyProposalMMSSRCNN(DistillProposalMMSSRCNN):
    def forward(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        """
        Arguments:
            images (list[Tensor] or ImageList): images to be processed
            targets (list[BoxList]): ground-truth boxes present in the image (optional)
                [or (list[ndarray]) with image-level labels in weakly supervised settings]

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        # Inference
        if not self.training:
            return self.inference(batched_inputs)

        # Language backbone
        targets = self.preprocess_text(batched_inputs)
        input_caption = self.language_backbone(targets)

        # Visual backbone
        images = self.preprocess_image(batched_inputs)
        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None

        features = self.backbone(images.tensor)

        # Run the RPN
        if self.proposal_generator is not None:
            proposals, proposal_losses = self.proposal_generator(
                images, features, gt_instances
            )
        else:
            assert "proposals" in batched_inputs[0]
            proposals = [x["proposals"].to(self.device) for x in batched_inputs]
            proposal_losses = {}

        # Normal detector
        _, box_features, box_proposals, detector_losses = self.roi_heads(
            images, features, proposals, gt_instances
        )

        if self.vis_period > 0:
            storage = get_event_storage()
            if storage.iter % self.vis_period == 0:
                self.visualize_training(batched_inputs, box_proposals)

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)

        # MMSS
        batch_size, _, image_h, image_w = images.tensor.shape
        self.log("images.tensor", images.tensor)

        # Grounding BoxMMSS
        num_boxes = [len(box) for box in box_proposals]
        num_boxes = min(num_boxes)
        if self.spatial_dropout > 0 and self.training:
            num_boxes = min(num_boxes, self.spatial_dropout)

        subsampled_box_features = []
        subsampled_box_proposals = []
        for i_box, box in enumerate(box_proposals):
            idx = np.arange(len(box))
            np.random.shuffle(idx)
            idx = idx[:num_boxes]
            subsampled_box_proposals.append(box[idx])
            subsampled_box_features.append(box_features[i_box][idx])

        if self.vis_period > 0:
            storage = get_event_storage()
            if storage.iter % self.vis_period == 0:
                self.visualize_training(batched_inputs, subsampled_box_proposals)

        # The box centers in a Nx2 array of (x, y).
        subsampled_box_centers = [
            prop.get("proposal_boxes").get_centers()
            for prop in subsampled_box_proposals
        ]
        # Normalize centers to [0,1]
        subsampled_box_loc = [
            torch.stack(
                [
                    center[:, 0] / prop._image_size[1],
                    center[:, 1] / prop._image_size[0],
                ],
                dim=-1,
            )
            for center, prop in zip(subsampled_box_centers, subsampled_box_proposals)
        ]

        flattened_box_features = torch.nn.utils.rnn.pad_sequence(
            subsampled_box_features, batch_first=True
        )
        flattened_box_loc = torch.nn.utils.rnn.pad_sequence(
            subsampled_box_loc, batch_first=True
        )
        flattened_box_mask = np.ones([batch_size, num_boxes], dtype=np.uint8)

        input_boxes = {
            "region_features": flattened_box_features,
            "region_mask": torch.tensor(flattened_box_mask).to(self.device),
            "region_loc": flattened_box_loc,
            "mvm_mask": torch.zeros(batch_size, num_boxes).to(self.device),
            "target_region_features": flattened_box_features,
        }
        if self.mvm:
            raise NotImplementedError

        mmss_outputs = {}
        mmss_losses = {}
        mmss_distributions = {}
        for head in self.mmss_heads:
            box_o, box_l, box_d = self.mmss_heads[head](input_boxes, input_caption)
            box_o = {"Box " + k: v for k, v in box_o.items()}
            box_l = {"Box " + k: v for k, v in box_l.items()}
            box_d = {"box_" + k: v for k, v in box_d.items()}
            mmss_outputs.update(box_o)
            mmss_losses.update(box_l)
            mmss_distributions.update(box_d)

        del input_boxes
        del input_caption

        # Perform distillation loss
        box_trans_pw_cost = mmss_distributions["box_trans"]
        box_w2r_pw_cost = mmss_distributions["box_w2r"]
        box_r2w_pw_cost = mmss_distributions["box_r2w"]
        box_kd_loss = self.distill_loss(
            box_trans_pw_cost, box_w2r_pw_cost, box_r2w_pw_cost
        )
        mmss_losses["box_kd_loss"] = box_kd_loss

        for v in mmss_losses.values():
            if torch.isnan(v):
                for head in self.mmss_heads:
                    print(head, self.mmss_heads[head].log_info)
                print(image_sizes, grid_sizes)
                raise ValueError()

        losses.update(mmss_losses)

        return mmss_outputs, losses
