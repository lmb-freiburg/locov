import os
import copy
import json
import torch
import random
import string
import logging
import numpy as np
import pandas as pd
from PIL import Image
from typing import Dict, List, Optional, Tuple, Union

from detectron2 import data
from detectron2.config import CfgNode
from detectron2.structures import BoxMode
from detectron2.config import configurable
from detectron2.data import transforms as T
from detectron2.data import detection_utils as utils

import ovr.data.detection_utils as wsog_utils

class BasicTextImageDatasetMapper(data.DatasetMapper):
    @configurable
    def __init__(
        self,
        is_train: bool,
        *,
        augmentations: List[Union[T.Augmentation, T.Transform]],
        train_aug,
        image_format: str,
        use_instance_mask: bool = False,
        use_keypoint: bool = False,
        instance_mask_format: str = "polygon",
        keypoint_hflip_indices: Optional[np.ndarray] = None,
        precomputed_proposal_topk: Optional[int] = None,
        recompute_boxes: bool = False,
    ):
        if recompute_boxes:
            assert use_instance_mask, "recompute_boxes requires instance masks"
        # fmt: off
        self.is_train               = is_train
        self.augmentations          = T.AugmentationList(augmentations)
        self.train_aug              = train_aug
        self.image_format           = image_format
        self.use_instance_mask      = use_instance_mask
        self.instance_mask_format   = instance_mask_format
        self.use_keypoint           = use_keypoint
        self.keypoint_hflip_indices = keypoint_hflip_indices
        self.proposal_topk          = precomputed_proposal_topk
        self.recompute_boxes        = recompute_boxes
        # fmt: on
        logger = logging.getLogger(__name__)
        mode = "training" if is_train else "inference"
        logger.info(f"[DatasetMapper] Augmentations used in {mode}: {augmentations}")
        if self.is_train:
            logger.info(f"[DatasetMapper] Additional augmentations used in train: {train_aug}")

    @classmethod
    def from_config(cls, cfg, is_train: bool = True):
        augs = utils.build_augmentation(cfg, is_train)
        if cfg.INPUT.CROP.ENABLED and is_train:
            augs.insert(0, T.RandomCrop(cfg.INPUT.CROP.TYPE, cfg.INPUT.CROP.SIZE))
            recompute_boxes = cfg.MODEL.MASK_ON
        else:
            recompute_boxes = False
        train_augs = wsog_utils.build_complete_augmentation(cfg, is_train)

        ret = {
            "is_train": is_train,
            "augmentations": augs,
            "train_aug": train_augs,
            "image_format": cfg.INPUT.FORMAT,
            "use_instance_mask": cfg.MODEL.MASK_ON,
            "instance_mask_format": cfg.INPUT.MASK_FORMAT,
            "use_keypoint": cfg.MODEL.KEYPOINT_ON,
            "recompute_boxes": recompute_boxes,
        }

        if cfg.MODEL.KEYPOINT_ON:
            ret["keypoint_hflip_indices"] = utils.create_keypoint_hflip_indices(cfg.DATASETS.TRAIN)

        if cfg.MODEL.LOAD_OBJ_PROPOSALS:
            ret["precomputed_proposal_topk"] = (
                cfg.DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TRAIN
                if is_train
                else cfg.DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TEST
            )
        return ret

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        # USER: Write your own image loading if it's not from a file
        loaded_image = False
        try:
            image = utils.read_image(dataset_dict["file_name"], format=self.image_format)
            loaded_image = True
        except:
            image = np.zeros((dataset_dict['height'], dataset_dict['width'], 3), dtype=np.uint8)
            print("Image not loaded {}, replaced by black image".format(dataset_dict["file_name"]))

        
        wsog_utils.check_image_size(dataset_dict, image)
        # USER: Remove if you don't do semantic/panoptic segmentation.
        if "sem_seg_file_name" in dataset_dict:
            sem_seg_gt = utils.read_image(dataset_dict.pop("sem_seg_file_name"), "L").squeeze(2)
        else:
            sem_seg_gt = None

        aug_input = T.AugInput(image, sem_seg=sem_seg_gt)
        transforms = self.augmentations(aug_input)
        image, sem_seg_gt = aug_input.image, aug_input.sem_seg

        image_shape = image.shape[:2]  # h, w
        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        if sem_seg_gt is not None:
            dataset_dict["sem_seg"] = torch.as_tensor(sem_seg_gt.astype("long"))

        # USER: Remove if you don't use pre-computed proposals.
        # Most users would not need this feature.
        if self.proposal_topk is not None:
            utils.transform_proposals(
                dataset_dict, image_shape, transforms, proposal_topk=self.proposal_topk
            )

        if not self.is_train:
            # USER: Modify this if you want to keep them for some reason.
            # dataset_dict.pop("annotations", None)
            dataset_dict.pop("sem_seg_file_name", None)
            # return dataset_dict

        if "annotations" in dataset_dict:
            # USER: Modify this if you want to keep them for some reason.
            for anno in dataset_dict["annotations"]:
                if not self.use_instance_mask:
                    anno.pop("segmentation", None)
                if not self.use_keypoint:
                    anno.pop("keypoints", None)

            # USER: Implement additional transformations if you have other types of data
            annos = [
                utils.transform_instance_annotations(
                    obj, transforms, image_shape, keypoint_hflip_indices=self.keypoint_hflip_indices
                )
                for obj in dataset_dict.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]
            instances = wsog_utils.annotations_to_instances(
                annos, image_shape, mask_format=self.instance_mask_format
            )

            # After transforms such as cropping are applied, the bounding box may no longer
            # tightly bound the object. As an example, imagine a triangle object
            # [(0,0), (2,0), (0,2)] cropped by a box [(1,0),(2,2)] (XYXY format). The tight
            # bounding box of the cropped triangle should be [(1,0),(2,1)], which is not equal to
            # the intersection of original bounding box and the cropping box.
            if self.recompute_boxes:
                instances.gt_boxes = instances.gt_masks.get_bounding_boxes()

            dataset_dict["instances"] = utils.filter_empty_instances(instances)
        
        if self.train_aug is not None and self.is_train:
            # apply strong augmentation
            # We use torchvision augmentation, which is not compatiable with
            # detectron2, which use numpy format for images. Thus, we need to
            # convert to PIL format first.
            image = dataset_dict["image"].numpy().transpose(1, 2, 0)
            image_pil = Image.fromarray(image.astype("uint8"), "RGB")
            image_train_aug = np.array(self.train_aug(image_pil))
            dataset_dict["image"] = torch.as_tensor(
                np.ascontiguousarray(image_train_aug.transpose(2, 0, 1)))

        # select caption
        if 'caption' in dataset_dict:
            if isinstance(dataset_dict['caption'], list):
                if self.is_train:
                    dataset_dict['caption'] = random.choice(dataset_dict['caption'])
                else:
                    dataset_dict['caption'] = dataset_dict['caption'][0]
                if not loaded_image:
                    dataset_dict['caption'] = "A black image."

        return dataset_dict


class TextImageDatasetMapperNoise(BasicTextImageDatasetMapper):
    def __init__(
        self,
        cfg,
        metadata,
        is_train: bool,
    ):
        super().__init__(cfg, is_train)
        self.noise_offline = cfg.INPUT.NOISE_OFFLINE
        self.noise_rm_box = cfg.INPUT.NOISE_RM_BBOX
        self.noise_cls = cfg.INPUT.NOISE_CLS
        self.noise_loc_bbox = cfg.INPUT.NOISE_LOC
        self.noise_bbox = cfg.INPUT.NOISE_BBOX
        self.noise_ign = cfg.INPUT.NOISE_IGN
        # self.binary_detection = cfg.INPUT.BINARY_DETECTION
        self.metadata = metadata

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept with additional aumentations
        """
        dataset_dict = copy.deepcopy(dataset_dict)
        if not self.noise_offline:
            if self.noise_rm_box>0:
                dataset_dict = wsog_utils.rm_annotation(dataset_dict, self.noise_rm_box)
            if self.noise_cls>0:
                if random.random()>self.noise_cls:
                    dataset_dict = wsog_utils.add_noise_cls(dataset_dict, self.metadata.thing_classes)
            if self.noise_loc_bbox>0:
                if random.random()>self.noise_loc_bbox:
                    dataset_dict = wsog_utils.add_noise_loc(dataset_dict, self.noise_loc_bbox)
            if self.noise_bbox>0:
                dataset_dict = wsog_utils.add_noise_annotation(dataset_dict, self.noise_bbox, self.metadata.thing_classes)

        if self.noise_ign>0:
            dataset_dict = wsog_utils.online_ign_annotation(dataset_dict, self.metadata.thing_classes)

        dataset_dict = super().__call__(dataset_dict)
        return dataset_dict