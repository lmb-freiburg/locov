# import os
# import sys
# sys.path.insert(0,os.getcwd())
from detectron2.data import DatasetCatalog, MetadataCatalog
from ovr.data.detection_utils import add_caption_and_category
from detectron2.data.datasets.coco import load_coco_json
import numpy as np
import json, os
import pickle

categories_seen = [
    {"id": 1, "name": "person"},
    {"id": 2, "name": "bicycle"},
    {"id": 3, "name": "car"},
    {"id": 4, "name": "motorcycle"},
    {"id": 7, "name": "train"},
    {"id": 8, "name": "truck"},
    {"id": 9, "name": "boat"},
    {"id": 15, "name": "bench"},
    {"id": 16, "name": "bird"},
    {"id": 19, "name": "horse"},
    {"id": 20, "name": "sheep"},
    {"id": 23, "name": "bear"},
    {"id": 24, "name": "zebra"},
    {"id": 25, "name": "giraffe"},
    {"id": 27, "name": "backpack"},
    {"id": 31, "name": "handbag"},
    {"id": 33, "name": "suitcase"},
    {"id": 34, "name": "frisbee"},
    {"id": 35, "name": "skis"},
    {"id": 38, "name": "kite"},
    {"id": 42, "name": "surfboard"},
    {"id": 44, "name": "bottle"},
    {"id": 48, "name": "fork"},
    {"id": 50, "name": "spoon"},
    {"id": 51, "name": "bowl"},
    {"id": 52, "name": "banana"},
    {"id": 53, "name": "apple"},
    {"id": 54, "name": "sandwich"},
    {"id": 55, "name": "orange"},
    {"id": 56, "name": "broccoli"},
    {"id": 57, "name": "carrot"},
    {"id": 59, "name": "pizza"},
    {"id": 60, "name": "donut"},
    {"id": 62, "name": "chair"},
    {"id": 65, "name": "bed"},
    {"id": 70, "name": "toilet"},
    {"id": 72, "name": "tv"},
    {"id": 73, "name": "laptop"},
    {"id": 74, "name": "mouse"},
    {"id": 75, "name": "remote"},
    {"id": 78, "name": "microwave"},
    {"id": 79, "name": "oven"},
    {"id": 80, "name": "toaster"},
    {"id": 82, "name": "refrigerator"},
    {"id": 84, "name": "book"},
    {"id": 85, "name": "clock"},
    {"id": 86, "name": "vase"},
    {"id": 90, "name": "toothbrush"},
]

# 17 out of 80
categories_unseen = [
    {"id": 5, "name": "airplane"},
    {"id": 6, "name": "bus"},
    {"id": 17, "name": "cat"},
    {"id": 18, "name": "dog"},
    {"id": 21, "name": "cow"},
    {"id": 22, "name": "elephant"},
    {"id": 28, "name": "umbrella"},
    {"id": 32, "name": "tie"},
    {"id": 36, "name": "snowboard"},
    {"id": 41, "name": "skateboard"},
    {"id": 47, "name": "cup"},
    {"id": 49, "name": "knife"},
    {"id": 61, "name": "cake"},
    {"id": 63, "name": "couch"},
    {"id": 76, "name": "keyboard"},
    {"id": 81, "name": "sink"},
    {"id": 87, "name": "scissors"},
]


COCO_DATASETS = {
    # Captions set
    "coco_captions_train": {
        "img_dir": "datasets_data/coco/train2017",
        "ann_file": "datasets_data/coco/annotations/instances_train2017.json",
        "cap_file": "datasets_data/coco/annotations/captions_train2017.json",
    },
    "coco_captions_val": {
        "img_dir": "datasets_data/coco/val2017",
        "ann_file": "datasets_data/coco/annotations/instances_val2017.json",
        "cap_file": "datasets_data/coco/annotations/captions_val2017.json",
    },
    "coco_captions_train_seen": {
        "img_dir": "datasets_data/coco/train2017",
        "ann_file": "datasets_data/zero-shot/coco/instances_train2017_seen_2.json",
        "cap_file": "datasets_data/coco/annotations/captions_train2017.json",
    },
    "coco_captions_val_seen": {
        "img_dir": "datasets_data/coco/val2017",
        "ann_file": "datasets_data/zero-shot/coco/instances_val2017_seen_2.json",
        "cap_file": "datasets_data/coco/annotations/captions_val2017.json",
    },
    # Captions with proposals set
    "coco_captions_train_proposals": {
        "img_dir": "datasets_data/coco/train2017",
        "ann_file": "datasets_data/coco/annotations/instances_train2017.json",
        "cap_file": "datasets_data/coco/annotations/captions_train2017.json",
        "obj_prop": "datasets_data/proposals/coco_train2017_voc.pkl",
    },
    "coco_captions_train_seen_proposals": {
        "img_dir": "datasets_data/coco/train2017",
        "ann_file": "datasets_data/coco/annotations/instances_train2017.json",
        "cap_file": "datasets_data/coco/annotations/captions_train2017.json",
        "obj_prop": "datasets_data/proposals/coco_train2017_seen.pkl",
    },
    # Object detection set for zero shot
    "coco_train": {
        "img_dir": "datasets_data/coco/train2017",
        "ann_file": "datasets_data/coco/annotations/instances_train2017.json",
    },
    "coco_zeroshot_train": {
        "img_dir": "datasets_data/coco/train2017",
        "ann_file": "datasets_data/zero-shot/coco/instances_train2017_seen_2.json",
    },
    "coco_zeroshot_val": {
        "img_dir": "datasets_data/coco/val2017",
        "ann_file": "datasets_data/zero-shot/coco/instances_val2017_unseen_2.json",
    },
    "coco_generalized_zeroshot_val": {
        "img_dir": "datasets_data/coco/val2017",
        "ann_file": "datasets_data/zero-shot/coco/instances_val2017_all_2.json",
        "cap_file": "datasets_data/coco/annotations/captions_val2017.json",
    },
    "coco_not_zeroshot_val": {
        "img_dir": "datasets_data/coco/val2017",
        "ann_file": "datasets_data/zero-shot/coco/instances_val2017_seen_2.json",
    },
    "coco_zeroshot_plus_unseen_train": {
        "img_dir": "datasets_data/coco/train2017",
        "ann_file": "datasets_data/zero-shot/coco/instances_train2017_all_2.json",
    },
    # Object detection set all
    "coco_2017_train": {
        "img_dir": "datasets_data/coco/train2017",
        "ann_file": "datasets_data/zero-shot/coco/instances_train2017_full.json",
        "cap_file": "datasets_data/coco/annotations/captions_train2017.json",
    },
    "coco_2017_val": {
        "img_dir": "datasets_data/coco/val2017",
        "ann_file": "datasets_data/zero-shot/coco/instances_val2017_full.json",
        "cap_file": "datasets_data/coco/annotations/captions_val2017.json",
    },
}


def register_coco_instances(
    name, metadata, json_file, image_root, extra_annotation_keys=None
):
    """
    Register a dataset in COCO's json annotation format for
    instance detection, instance segmentation and keypoint detection.
    (i.e., Type 1 and 2 in http://cocodataset.org/#format-data.
    `instances*.json` and `person_keypoints*.json` in the dataset).

    This is an example of how to register a new dataset.
    You can do something similar to this function, to register new datasets.

    Args:
        name (str): the name that identifies a dataset, e.g. "coco_2014_train".
        metadata (dict): extra metadata associated with this dataset.  You can
            leave it as an empty dict.
        json_file (str): path to the json instance annotation file.
        image_root (str or path-like): directory which contains all the images.
    """
    assert isinstance(name, str), name
    assert isinstance(json_file, (str, os.PathLike)), json_file
    assert isinstance(image_root, (str, os.PathLike)), image_root
    # 1. register a function which returns dicts
    DatasetCatalog.register(
        name, lambda: load_coco_json(json_file, image_root, name, extra_annotation_keys)
    )

    # 2. Optionally, add metadata about this dataset,
    # since they might be useful in evaluation, visualization or logging
    MetadataCatalog.get(name).set(
        json_file=json_file, image_root=image_root, evaluator_type="coco", **metadata
    )


def register_dataset(dataset_name):
    metadata = {}
    if dataset_name not in COCO_DATASETS.keys():
        raise NotImplementedError("Not paths for dataset " + dataset_name)

    dataset_paths = COCO_DATASETS[dataset_name]

    if dataset_name not in DatasetCatalog and dataset_name not in MetadataCatalog:
        extra_annotation_keys = ["segmentation", "area", "id"]
        register_coco_instances(
            dataset_name,
            metadata,
            dataset_paths["ann_file"],
            dataset_paths["img_dir"],
            extra_annotation_keys,
        )

    # Need to initialized to set thing_classes in dataset metadata
    dataset = DatasetCatalog.get(dataset_name)
    dataset_metadata = MetadataCatalog.get(dataset_name)
    if "cap_file" in dataset_paths.keys():
        print("Adding captions for " + dataset_name)
        captions_file = json.load(open(dataset_paths["cap_file"], "r"))
        images = captions_file["images"]
        annotations = captions_file["annotations"]

        captions_dict = {}
        for ann in annotations:
            if ann["image_id"] not in captions_dict.keys():
                captions_dict[ann["image_id"]] = [ann["caption"]]
            else:
                captions_dict[ann["image_id"]].append(ann["caption"])

        dataset_metadata.set(captions_dict=captions_dict)

    # Add noun embeddings
    if "obj_file" in dataset_paths.keys():
        noun_emb_file = dataset_paths["obj_file"]
    else:
        noun_emb_file = "datasets_data/embeddings/coco_nouns_bertemb.json"
    print("Adding embeddings for " + dataset_name)
    with open(noun_emb_file, "r") as fin:
        noun_embeddings = json.load(fin)

    class_embeddings = {}
    emb_dim = len(noun_embeddings[list(noun_embeddings.keys())[0]])
    # Adding background class
    class_emb_mtx = np.zeros(
        (len(dataset_metadata.thing_classes) + 1, emb_dim), dtype=np.float32
    )

    save_dict = False
    for idx, noun in enumerate(dataset_metadata.thing_classes):
        class_embeddings[idx] = np.asarray(noun_embeddings[noun], dtype=np.float32)
        if len(class_embeddings[idx].shape) == 1:
            class_emb_mtx[idx, :] = class_embeddings[idx]
        else:
            save_dict = True

    if save_dict:
        dataset_metadata.set(class_embeddings=class_embeddings)
    dataset_metadata.set(class_emb_mtx=class_emb_mtx)

    # Add object generic proposals
    if "obj_prop" in dataset_paths.keys():
        print("Adding object proposals for " + dataset_name)
        with open(dataset_paths["obj_prop"], "rb") as fin:
            object_proposals = pickle.load(fin, encoding="latin1")
        dict_object_proposals = {sample[0]: sample[1] for sample in object_proposals}
        dataset_metadata.set(object_proposals=dict_object_proposals)

    return


if __name__ == "__main__":
    """
    Test COCO dataset loader.

    Usage:
        "dataset_name" can be "mistates_train"
    """
    dataset_name = "coco_zeroshot_train"

    register_dataset(dataset_name)
    meta = MetadataCatalog.get(dataset_name)
    samples = DatasetCatalog.get(dataset_name)
    import ipdb

    ipdb.set_trace()
