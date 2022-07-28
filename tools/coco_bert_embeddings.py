import os
import sys

sys.path.insert(0, os.getcwd())
import json
import numpy as np
import torch

from detectron2.config import get_cfg
from ovr.config.config import add_ovr_config
from ovr.modeling.language.transf_models import BERT

cfg = get_cfg()
add_ovr_config(cfg)

bert = BERT(cfg)
_ = bert.to("cuda")

with open("datasets_data/coco/annotations/instances_train2017.json", "r") as fin:
    coco_train_anno_all = json.load(fin)

class_list = [category["name"] for category in coco_train_anno_all["categories"]]

encoded_class_list = bert(class_list)

mask = (1 - encoded_class_list["special_tokens_mask"]).to(torch.float32)

embeddings = (encoded_class_list["input_embeddings"] * mask[:, :, None]).sum(
    1
) / mask.sum(1)[:, None]

class_name_to_bertemb = {}
for c, emb in zip(class_list, embeddings.tolist()):
    class_name_to_bertemb[c] = emb

file_emb = "datasets_data/embeddings/coco_nouns_bertemb.json"
with open(file_emb, "w") as fout:
    json.dump(class_name_to_bertemb, fout)
    print("Embeddings saved {}".format(file_emb))

# with open(file_emb, 'r') as fin:
#     coco_nouns_bertemb = json.load(fin)
