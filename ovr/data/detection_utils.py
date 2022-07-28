import copy
import torch
import random
import numpy as np
import torchvision.transforms as transforms
from detectron2.data import transforms as T
from detectron2.structures import (
    BitMasks,
    Boxes,
    BoxMode,
    Instances,
    Keypoints,
    PolygonMasks,
    RotatedBoxes,
    polygons_to_bitmask,
)
from ovr.data.transforms.augmentation_impl import (
    GaussianBlur,
)

def check_image_size(dataset_dict, image):
    """
    Raise an error if the image does not match the size specified in the dict.
    """
    if "width" in dataset_dict or "height" in dataset_dict:
        image_wh = (image.shape[1], image.shape[0])
        expected_wh = (dataset_dict["width"], dataset_dict["height"])
        if not image_wh == expected_wh:
            if image_wh == (expected_wh[1], expected_wh[0]):
                dataset_dict["width"] = expected_wh[1]
                dataset_dict["height"] = expected_wh[0]
            else:
                print("Mismatched image shape{}, got {}, expect {}.".format(
                    " for image " + dataset_dict["file_name"]
                    if "file_name" in dataset_dict
                    else "",
                    image_wh,
                    expected_wh,
                )
                + " Please check the width/height in your annotation.")
                dataset_dict["width"] = image.shape[1]
                dataset_dict["height"] = image.shape[0]
            # raise SizeMismatchError(
            #     "Mismatched image shape{}, got {}, expect {}.".format(
            #         " for image " + dataset_dict["file_name"]
            #         if "file_name" in dataset_dict
            #         else "",
            #         image_wh,
            #         expected_wh,
            #     )
            #     + " Please check the width/height in your annotation."
            # )

    # To ensure bbox always remap to original image size
    if "width" not in dataset_dict:
        dataset_dict["width"] = image.shape[1]
    if "height" not in dataset_dict:
        dataset_dict["height"] = image.shape[0]

def build_complete_augmentation(cfg, is_train):
    """
    Create a list of :class:`Augmentation` from config.
    Now it includes resizing and flipping.

    Returns:
        list[Augmentation]
    """
    augmentation = []
    if is_train:
        # This is simialr to SimCLR https://arxiv.org/abs/2002.05709
        if cfg.INPUT.COLOR_JITTER>0:
            cj = cfg.INPUT.COLOR_JITTER
            augmentation.append(
                transforms.RandomApply([transforms.ColorJitter(cj, cj, cj, 0.1)], p=0.8)
            )
        if cfg.INPUT.RANDOM_GRAY_SCALE:
            augmentation.append(transforms.RandomGrayscale(p=0.2))
        if cfg.INPUT.GAUSSIAN_BLUR:
            augmentation.append(transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=0.5))
        if cfg.INPUT.RANDOM_ERASE:
            randcrop_transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.RandomErasing(
                        p=0.7, scale=(0.05, 0.2), ratio=(0.3, 3.3), value="random"
                    ),
                    transforms.RandomErasing(
                        p=0.5, scale=(0.02, 0.2), ratio=(0.1, 6), value="random"
                    ),
                    transforms.RandomErasing(
                        p=0.3, scale=(0.02, 0.2), ratio=(0.05, 8), value="random"
                    ),
                    transforms.ToPILImage(),
                ]
            )
            augmentation.append(randcrop_transform)

    if len(augmentation)>0:
        return transforms.Compose(augmentation)
    return None

# Class verification when evaluating 

# noise additions
def add_noise_annotation(sample_dict, noise_bbox, thing_classes):
    """
    Helper function to create noisy annotations
    """
    record = copy.deepcopy(sample_dict)
    height, width = sample_dict['height'], sample_dict['width']
    annotations = sample_dict["annotations"]
    if noise_bbox<1 and noise_bbox>0:
        n_boxes = int(noise_bbox*len(annotations))
    else:
        n_boxes = int(noise_bbox)
    for _ in range(n_boxes):
        # format for training = [xmin, ymin, x_w, y_h] in [0, w]x[0, h]
        # build random bounding box
        x_w = random.randint(width//6, width*4//6)
        xmin = random.randint(0, width-x_w-1)
        y_h = random.randint(height//6, height*4//6)
        ymin = random.randint(0, height-y_h-1)
        category_id = random.randint(0, len(thing_classes)-1)

        obj = {
            "bbox": [xmin, ymin, x_w, y_h],
            "bbox_mode": BoxMode.XYWH_ABS,
            "category_id": category_id,
            "iscrowd": 0,
            "image_id": sample_dict['image_id'],
            "category": thing_classes[category_id],
        }
        annotations.append(obj)
    record["annotations"] = annotations
    return record

def add_noise_cls(sample_dict, thing_classes):
    """
    Helper function to create noisy annotations
    """
    record = copy.deepcopy(sample_dict)
    for ann in record["annotations"]:
        category_id = random.randint(0, len(thing_classes)-1)
        ann["category_id"] = category_id
        ann["category"] = thing_classes[category_id]
    return record

def rm_annotation(sample_dict, noise_rm_box):
    """
    Helper function to create noisy annotations
    """
    record = copy.deepcopy(sample_dict)
    height, width = sample_dict['height'], sample_dict['width']
    n_keep = int((1-noise_rm_box)*len(record["annotations"]))
    if n_keep<1:
        # keep at least one bbox
        return record
    idx_keep = random.sample(range(len(record["annotations"])), n_keep)
    annotations = [ann for idx_ann, ann in enumerate(record["annotations"]) if idx_ann in idx_keep]
    record["annotations"] = annotations
    return record

def ign_annotation(sample_dict, noise_ign_box, thing_classes):
    """
    Helper function to create noisy annotations
    """
    record = copy.deepcopy(sample_dict)
    n_keep = int((1-noise_ign_box)*len(record["annotations"]))
    if n_keep<1:
        # keep at least one bbox
        return record
    idx_keep = random.sample(range(len(record["annotations"])), n_keep)
    annotations = []
    for idx_ann, ann in enumerate(record["annotations"]):
        if idx_ann in idx_keep:
            category_id = ann["category_id"]
            ann["category"] = thing_classes[category_id]
            annotations.append(ann)
        else:
            # ignore label by setting it to len(thing_classes)
            category_id = ann["category_id"]
            ann["category"] = thing_classes[category_id]
            ann["category_id"] = len(thing_classes)
            annotations.append(ann)
    record["annotations"] = annotations
    return record

def online_ign_annotation(sample_dict, thing_classes):
    """
    Helper function to create noisy annotations
    """
    record = copy.deepcopy(sample_dict)
    for idx_ann, ann in enumerate(record["annotations"]):
        if thing_classes[ann["category_id"]]=='ignore':
            ann["category_id"] = -1
    return record

def add_noise_loc(sample_dict, noise_loc):
    """
    Helper function to create noisy annotations
    """
    record = copy.deepcopy(sample_dict)
    height, width = sample_dict['height'], sample_dict['width']
    for ann in record["annotations"]:
        # format for training = [xmin, ymin, x_w, y_h] in [0, w]x[0, h]
        # shift randomly the bounding box
        o_bbox = ann["bbox"]
        xmin = max(o_bbox[0] + random.randint(-o_bbox[2]//8, o_bbox[2]//8), 0)
        ymin = max(o_bbox[1] + random.randint(-o_bbox[3]//8, o_bbox[3]//8), 0)
        x_w = min(o_bbox[2] + random.randint(-o_bbox[2]//8, o_bbox[2]//8), width-1)
        y_h = min(o_bbox[3] + random.randint(-o_bbox[3]//8, o_bbox[3]//8), height-1)
        ann["bbox"] = [xmin, ymin, x_w, y_h]
    return record

def correct_indexing(sample_dict, noun2code, thing_classes):
    """
    Helper function to correct indexing of classes
    """
    record = copy.deepcopy(sample_dict)
    for ann in record["annotations"]:
        category = thing_classes[ann["category_id"]]
        category_id = noun2code[category][0]
        ann["category_id"] = category_id
        ann["category"] = category
    return record


def add_caption_and_category(sample_dict, thing_classes, captions_dict):
    """
    Helper function to correct indexing of classes
    """
    record = copy.deepcopy(sample_dict)
    if sample_dict['image_id'] in captions_dict.keys():
        record['caption'] = ' '.join(cap+'.' if cap[-1]!='.' else cap for cap in captions_dict[sample_dict['image_id']])
    else:
        record['caption'] = ''
    for ann in record["annotations"]:
        category = thing_classes[ann["category_id"]]
        ann["category"] = category
    return record

def correct_indexing_and_specific_classes(sample_dict, noun2code, thing_classes, valid_classes):
    """
    Helper function to correct indexing of classes
    """
    record = copy.deepcopy(sample_dict)
    new_ann = []
    for ann in record["annotations"]:
        category = thing_classes[ann["category_id"]]
        category_id = noun2code[category][0]
        if category in valid_classes:
            ann["category_id"] = category_id
            ann["category"] = category
            new_ann.append(ann)
    record["annotations"] = new_ann
    return record


def binary_cls(sample_dict, thing_classes):
    """
    Helper function to create binary annotations
    """
    record = copy.deepcopy(sample_dict)
    for ann in record["annotations"]:
        category_id = ann["category_id"]
        ann["category_id"] = 1
        ann["category"] = thing_classes[category_id]
    return record



def annotations_to_instances(annos, image_size, mask_format="polygon"):
    """
    Create an :class:`Instances` object used by the models,
    from instance annotations in the dataset dict.

    Args:
        annos (list[dict]): a list of instance annotations in one image, each
            element for one instance.
        image_size (tuple): height, width

    Returns:
        Instances:
            It will contain fields "gt_boxes", "gt_classes",
            "gt_masks", "gt_keypoints", if they can be obtained from `annos`.
            This is the format that builtin models expect.
    """
    boxes = [BoxMode.convert(obj["bbox"], obj["bbox_mode"], BoxMode.XYXY_ABS) for obj in annos]
    target = Instances(image_size)
    target.gt_boxes = Boxes(boxes)

    classes = [int(obj["category_id"]) for obj in annos]
    classes = torch.tensor(classes, dtype=torch.int64)
    target.gt_classes = classes

    if len(annos) and "segmentation" in annos[0]:
        segms = [obj["segmentation"] for obj in annos]
        if mask_format == "polygon":
            try:
                masks = PolygonMasks(segms)
            except ValueError as e:
                raise ValueError(
                    "Failed to use mask_format=='polygon' from the given annotations!"
                ) from e
        else:
            assert mask_format == "bitmask", mask_format
            masks = []
            for segm in segms:
                if isinstance(segm, list):
                    # polygon
                    masks.append(polygons_to_bitmask(segm, *image_size))
                elif isinstance(segm, dict):
                    # COCO RLE
                    masks.append(mask_util.decode(segm))
                elif isinstance(segm, np.ndarray):
                    assert segm.ndim == 2, "Expect segmentation of 2 dimensions, got {}.".format(
                        segm.ndim
                    )
                    # mask array
                    masks.append(segm)
                else:
                    raise ValueError(
                        "Cannot convert segmentation of type '{}' to BitMasks!"
                        "Supported types are: polygons as list[list[float] or ndarray],"
                        " COCO-style RLE as a dict, or a binary segmentation mask "
                        " in a 2D numpy array of shape HxW.".format(type(segm))
                    )
            # torch.from_numpy does not support array with negative stride.
            masks = BitMasks(
                torch.stack([torch.from_numpy(np.ascontiguousarray(x)) for x in masks])
            )
        target.gt_masks = masks

    if len(annos) and "keypoints" in annos[0]:
        kpts = [obj.get("keypoints", []) for obj in annos]
        target.gt_keypoints = Keypoints(kpts)

    # Add other fields if there are more to add
    if len(annos):
        common_used_fields = {'iscrowd', 'bbox', 'category_id', 'area', 'bbox_mode'}
        anns_filds = set(annos[0].keys()).difference(common_used_fields)
        missing_filds_dict = {k: [ann[k] for ann in annos] for k in anns_filds}
        for k, v in missing_filds_dict.items():
            if isinstance(v, list):
                if isinstance(v[0], int):
                    v = torch.tensor(v, dtype=torch.int64)
                if isinstance(v[0], np.ndarray):
                    v = torch.stack([torch.from_numpy(np.ascontiguousarray(x)) for x in v])
            if torch.is_tensor(v):
                target.set(k,v)    

    return target