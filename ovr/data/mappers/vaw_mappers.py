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
from ovr.data.mappers.coco_mappers import change_proposals_as_gt

# VAW mapper
class VawImageDatasetMapper(BasicTextImageDatasetMapper):
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

        # add caption if it is contained in the dataset_dict
        if "caption" in dataset_dict.keys():
            nouns = []
            nouns_id = []
            for ann in dataset_dict["annotations"]:
                category_id = ann["category_id"]
                ann["category"] = self.metadata.thing_classes[category_id]
                nouns.append(ann["category"])
                nouns_id.append(category_id)
            dataset_dict["nouns"] = nouns
            dataset_dict["nouns_id"] = nouns_id

            if self.is_train and len(dataset_dict["caption"]) > 0:
                k = max(1, min(len(dataset_dict["caption"]), random.choice([2, 3, 4])))
                dataset_dict["caption"] = " ".join(
                    random.choices(dataset_dict["caption"], k=k)
                )
            elif not self.is_train and len(dataset_dict["caption"]) > 0:
                dataset_dict["caption"] = " ".join(dataset_dict["caption"])
            else:
                dataset_dict[
                    "caption"
                ] = "An image with no caption. Objects present " + " ".join(
                    dataset_dict["nouns"]
                )

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
