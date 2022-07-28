# Following OVR-cnn code
# https://github.com/alirezazareian/ovr-cnn/blob/master/ipynb/003.ipynb
import os
import sys

sys.path.insert(0, os.getcwd())
import json
import numpy as np
import torch
import copy

from ovr.data.datasets.coco_instances import categories_seen, categories_unseen

with open("datasets_data/coco/annotations/instances_train2017.json", "r") as fin:
    coco_train_anno_all = json.load(fin)
coco_train_anno_seen = copy.deepcopy(coco_train_anno_all)
coco_train_anno_unseen = copy.deepcopy(coco_train_anno_all)

with open("datasets_data/coco/annotations/instances_val2017.json", "r") as fin:
    coco_val_anno_all = json.load(fin)
coco_val_anno_seen = copy.deepcopy(coco_val_anno_all)
coco_val_anno_unseen = copy.deepcopy(coco_val_anno_all)

print("Number of base categories", len(categories_seen))
print("Number of novel categories", len(categories_unseen))

categories_all = [item["name"] for item in coco_val_anno_all["categories"]]

split_name_list = {
    "seen": [obj["name"] for obj in categories_seen],
    "unseen": [obj["name"] for obj in categories_unseen],
}

class_id_to_split = {}
class_name_to_split = {}
for item in coco_val_anno_all["categories"]:
    if item["name"] in split_name_list["seen"]:
        class_id_to_split[item["id"]] = "seen"
        class_name_to_split[item["name"]] = "seen"
    elif item["name"] in split_name_list["unseen"]:
        class_id_to_split[item["id"]] = "unseen"
        class_name_to_split[item["name"]] = "unseen"


def filter_annotation(anno_dict, split_list):
    filtered_categories = []
    for item in anno_dict["categories"]:
        for split_name in split_list:
            if item["name"] in split_name_list[split_name]:
                item["split"] = split_name
                filtered_categories.append(item)
    anno_dict["categories"] = filtered_categories

    filtered_images = []
    filtered_annotations = []
    useful_image_ids = set()
    for item in anno_dict["annotations"]:
        if class_id_to_split.get(item["category_id"]) in split_list:
            filtered_annotations.append(item)
            useful_image_ids.add(item["image_id"])
    for item in anno_dict["images"]:
        if item["id"] in useful_image_ids:
            filtered_images.append(item)
    anno_dict["annotations"] = filtered_annotations
    anno_dict["images"] = filtered_images
    return anno_dict


coco_train_anno_seen = filter_annotation(coco_train_anno_seen, ["seen"])
coco_train_anno_unseen = filter_annotation(coco_train_anno_unseen, ["unseen"])
filter_annotation(coco_train_anno_all, ["seen", "unseen"])

filter_annotation(coco_val_anno_seen, ["seen"])
filter_annotation(coco_val_anno_unseen, ["unseen"])
filter_annotation(coco_val_anno_all, ["seen", "unseen"])


print(
    "seen",
    len(coco_val_anno_seen["categories"]),
    len(coco_val_anno_seen["annotations"]),
)
print(
    "unseen",
    len(coco_val_anno_unseen["categories"]),
    len(coco_val_anno_unseen["annotations"]),
)
print(
    "seen+unseen",
    len(coco_val_anno_all["categories"]),
    len(coco_val_anno_all["annotations"]),
)

annotations_dict = "datasets_data/zero-shot/coco/"
os.makedirs(annotations_dict, exist_ok=True)
with open(annotations_dict + "/instances_train2017_seen_2.json", "w") as fout:
    json.dump(coco_train_anno_seen, fout)
with open(annotations_dict + "/instances_train2017_unseen_2.json", "w") as fout:
    json.dump(coco_train_anno_unseen, fout)
with open(annotations_dict + "/instances_train2017_all_2.json", "w") as fout:
    json.dump(coco_train_anno_all, fout)

with open(annotations_dict + "/instances_val2017_seen_2.json", "w") as fout:
    json.dump(coco_val_anno_seen, fout)
with open(annotations_dict + "/instances_val2017_unseen_2.json", "w") as fout:
    json.dump(coco_val_anno_unseen, fout)
with open(annotations_dict + "/instances_val2017_all_2.json", "w") as fout:
    json.dump(coco_val_anno_all, fout)
