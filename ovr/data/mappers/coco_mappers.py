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
from ovr.data.mappers.basic_mappers import BasicTextImageDatasetMapper

# Coco mapper
class CocoImageDatasetMapper(BasicTextImageDatasetMapper):
    def __init__(
        self,
        cfg,
        metadata,
        is_train: bool,
    ):
        super().__init__(cfg, is_train)
        self.metadata = metadata

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): dict of one sample images.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)

        # add caption if saved in metadata
        if self.metadata.get("captions_dict"):
            captions_dict = self.metadata.get("captions_dict")
            if dataset_dict["image_id"] in captions_dict.keys():
                if self.is_train:
                    dataset_dict["caption"] = random.choice(
                        captions_dict[dataset_dict["image_id"]]
                    )
                else:
                    dataset_dict["caption"] = captions_dict[dataset_dict["image_id"]][0]
                nouns = []
                nouns_id = []
                for ann in dataset_dict["annotations"]:
                    category_id = ann["category_id"]
                    ann["category"] = self.metadata.thing_classes[category_id]
                    nouns.append(ann["category"])
                    nouns_id.append(category_id)
                dataset_dict["nouns"] = nouns
                dataset_dict["nouns_id"] = nouns_id
            else:
                dataset_dict["caption"] = ""
                dataset_dict["nouns"] = []
                dataset_dict["nouns_id"] = []

        # add prop if saved in metadata
        if self.metadata.get("object_proposals"):
            proposals_dict = self.metadata.get("object_proposals")
            if dataset_dict["image_id"] in proposals_dict.keys():
                proposals = proposals_dict[dataset_dict["image_id"]]
                if isinstance(proposals, list):
                    proposals = proposals[0]
                dataset_dict["proposal_boxes"] = proposals[:, :4]
                dataset_dict["proposal_objectness_logits"] = proposals[:, 4]
                dataset_dict["proposal_bbox_mode"] = BoxMode.XYXY_ABS

        dataset_dict = super().__call__(dataset_dict)

        # replace gt_intances for obj_proposals, and set labels to binary
        if self.metadata.get("object_proposals"):
            dataset_dict = change_proposals_as_gt(dataset_dict)

        return dataset_dict


def change_proposals_as_gt(dataset_dict, objectness_thr=0.7, max_n_prop=200):
    dataset_dict = copy.deepcopy(dataset_dict)

    proposals = dataset_dict.pop("proposals")
    instances = dataset_dict.pop("instances")
    mask_valid = proposals.get("objectness_logits") > objectness_thr
    # if mask_valid.sum()>max_n_prop and len(mask_valid)>2*max_n_prop:
    #     mask_valid[1:2*max_n_prop+1:2] = False
    #     if mask_valid.sum()>max_n_prop:
    #         mask_valid[2*max_n_prop+1:] = False
    save_instances = copy.deepcopy(proposals[mask_valid])
    save_instances.set("gt_classes", torch.ones(len(save_instances), dtype=torch.long))
    save_instances.set("gt_boxes", save_instances.get("proposal_boxes"))
    save_instances.remove("proposal_boxes")

    dataset_dict["gt_obj"] = instances
    dataset_dict["instances"] = save_instances

    return dataset_dict
