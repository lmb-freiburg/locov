import contextlib
import copy
import os
import io
import json
import logging
import numpy as np
import dill as pickle
from fvcore.common.timer import Timer

from detectron2.structures import BoxMode
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.file_io import PathManager
from detectron2.data.datasets.lvis import load_lvis_json
from detectron2.data.datasets.lvis_v0_5_categories import LVIS_CATEGORIES as LVIS_V0_5_CATEGORIES
from detectron2.data.datasets.lvis_v1_categories import LVIS_CATEGORIES as LVIS_V1_CATEGORIES
from detectron2.data.datasets.builtin_meta import COCO_CATEGORIES

LVIS_DATASETS = {
    "common_dict": {
        "img_dir": "datasets_data/coco/",
        "cap_file": "datasets_data/coco/annotations/captions_*2017.json",
        "nar_file": "datasets_data/localized_narratives/coco_*",
    },
    "common_train_dict": {
        "ann_file": "datasets_data/lvis/lvis_v1_train.json",
    },
    "common_val_dict": {
        "ann_file": "datasets_data/lvis/lvis_v1_val.json",
    },
    "lvis_v1_caption_train_proposals":{
        "obj_prop": "datasets_data/proposals/coco_train2017_seen.pkl",
        "obj_file": "datasets_data/embeddings/lvis_v1_nouns_bertemb.json"
    },
    "lvis_v1_caption_train":{
        "obj_file": "datasets_data/embeddings/lvis_v1_nouns_bertemb.json"
    },
    "lvis_v1_caption_val":{
        "obj_file": "datasets_data/embeddings/lvis_v1_nouns_bertemb.json"
    },
    "lvis_instance_v1_train":{
    },
    "lvis_instance_v1_val":{
    },
    "lvis_v1_all_train":{
        "obj_file": "datasets_data/embeddings/lvis_v1_nouns_bertemb.json"
    },
    "lvis_v1_base_train":{
        "obj_set": ["c", "f"],
        "obj_file": "datasets_data/embeddings/lvis_v1_nouns_bertemb.json"
    },
    "lvis_v1_generalized_val":{
        "obj_set": ["all"],
        "obj_file": "datasets_data/embeddings/lvis_v1_nouns_bertemb.json"
    },
    "lvis_v1_novel_val":{
        "obj_set": ["r"],
        "obj_file": "datasets_data/embeddings/lvis_v1_nouns_bertemb.json"
    },
    "lvis_v1_base_val":{
        "obj_set": ["c", "f"],
        "obj_file": "datasets_data/embeddings/lvis_v1_nouns_bertemb.json"
    }
}

logger = logging.getLogger(name=__name__)

__all__ = ["register_lvis_instances",]

def get_lvis_instances_meta_set(dataset_name):
    if "cocofied" in dataset_name:
        assert len(COCO_CATEGORIES) == 133
        CATEGORIES = COCO_CATEGORIES
    elif "v0.5" in dataset_name:
        assert len(LVIS_V0_5_CATEGORIES) == 1230
        CATEGORIES = LVIS_V0_5_CATEGORIES
    elif "v1" in dataset_name:
        assert len(LVIS_V1_CATEGORIES) == 1203
        CATEGORIES = LVIS_V1_CATEGORIES
    else:
        raise ValueError("No built-in metadata for dataset {}".format(dataset_name))
    
    cat_ids = [k["id"] for k in CATEGORIES]
    assert min(cat_ids) == 1 and max(cat_ids) == len(
        cat_ids
    ), "Category ids are not in [1, #categories], as expected"
    lvis_categories = sorted(CATEGORIES, key=lambda x: x["id"])
    thing_classes = []
    freq_classes = {}
    dict_classes = {}
    for k in lvis_categories:
        class_name = k["synonyms"][0]
        thing_classes.append(class_name)
        dict_classes[class_name] = k
        if 'frequency' in k.keys():
            if k['frequency'] not in freq_classes.keys():
                freq_classes[k['frequency']] = []
            freq_classes[k['frequency']].append(class_name)
    meta = {
        "thing_classes": thing_classes,
        "freq_classes": freq_classes,
        "dict_classes": dict_classes
    }
    return meta

def update_lvis_image_dict(img_dict, select_cat=False, add_cap=False, add_prop=False, 
    metadata_default=None, categories_consider=None, cap_anns=None, dict_object_proposals=None):
    img_dict = copy.deepcopy(img_dict)
    if select_cat:
        new_anns = []
        for ann in img_dict["annotations"]:
            category_id = ann['category_id']
            category = metadata_default.thing_classes[category_id]
            if category in categories_consider:
                ann['category_id'] = metadata_default.cat2idx[category]
                new_anns.append(ann)
        img_dict["annotations"] = new_anns
    if add_cap:
        img_dict["caption"] = [cap['caption'] for cap in cap_anns[img_dict['image_id']]]
    if add_prop:
        proposals = dict_object_proposals.get(img_dict['image_id'])
        if proposals is not None:
            if isinstance(proposals, list):
                proposals = proposals[0]
            img_dict["proposal_boxes"] = proposals[:,:4]
            img_dict["proposal_objectness_logits"] = proposals[:,4]
            img_dict["proposal_bbox_mode"] = BoxMode.XYXY_ABS
    return img_dict

def register_lvis_instances(name, json_file, image_root, extra_annotation_keys=None, **kwargs):
    """
    Create dataset dicts for coco instances, by
    merging two dicts using "file_name" field to match their entries.

    Args:
        detection_dicts (list[dict]): lists of dicts for object detection or instance segmentation.
        sem_seg_dicts (list[dict]): lists of dicts for semantic segmentation.

    Returns:
        list[dict] (one per input image): Each dict contains all (key, value) pairs from dicts in
            both detection_dicts and sem_seg_dicts that correspond to the same image.
            The function assumes that the same key in different dicts has the same value.
    """
    
    assert isinstance(name, str), name
    assert isinstance(json_file, (str, os.PathLike)), json_file
    assert isinstance(image_root, (str, os.PathLike)), image_root
    logger.info("Loading dicts for {}.".format(name))  

    # 1. build default dataset
    logger.info("Loading annotations, file {}.".format(json_file))  
    timer = Timer()
    data_dicts = load_lvis_json(json_file, image_root, name, extra_annotation_keys)
    if timer.seconds() > 1:
        logger.info("Loading annotations takes {:.2f} seconds.".format(timer.seconds()))

    # clear data metadata to built it in the correct way
    metadata_default = MetadataCatalog.get(name)
    MetadataCatalog.remove(name)
    metadata = MetadataCatalog.get(name).set(image_root=image_root, json_file=json_file)
    update_args = {}

    # 2. load captions if needed
    cap_file = kwargs.get("cap_file")
    if cap_file is not None:
        logger.info("Loading captions, file {}.".format(cap_file))   
        from pycocotools.coco import COCO
        timer = Timer()
        if '*' in cap_file:
            cap_anns = {}
            for file in os.listdir(os.path.dirname(cap_file)):
                if file.startswith(os.path.basename(cap_file).split('*')[0]) and file.endswith(os.path.basename(cap_file).split('*')[1]):
                    part_cap_file = os.path.join(os.path.dirname(cap_file), file)
                    logger.info("--- Captions file {}.".format(part_cap_file)) 
                    part_cap_file = PathManager.get_local_path(part_cap_file)
                    with contextlib.redirect_stdout(io.StringIO()):
                        coco_api_cap = COCO(part_cap_file)
                    img_ids = sorted(coco_api_cap.imgs.keys())
                    cap_anns_part = {img_id:coco_api_cap.imgToAnns[img_id] for img_id in img_ids}
                    cap_anns.update(cap_anns_part)
        else:
            cap_file = PathManager.get_local_path(cap_file)
            with contextlib.redirect_stdout(io.StringIO()):
                coco_api_cap = COCO(cap_file)
            img_ids = sorted(coco_api_cap.imgs.keys())
            cap_anns = {img_id:coco_api_cap.imgToAnns[img_id] for img_id in img_ids}
        
        if timer.seconds() > 1:
            logger.info("Loading captions takes {:.2f} seconds.".format(timer.seconds()))
        add_cap = True
        metadata.set(cap_file=cap_file)
        update_args["cap_anns"] = cap_anns
    else:
        add_cap = False
    update_args["add_cap"] = add_cap

    # 3. Filter out instances if the set should only contain specific categories
    obj_set = kwargs.get("obj_set")
    if obj_set is not None:
        logger.info("Filter out categories, object set {}.".format(obj_set))   
        # select the categories to consider
        meta_categories = get_lvis_instances_meta_set(name)
        categories_consider = set()
        for set_name in obj_set:
            if set_name in {'r', 'c', 'f'}:
                categories_consider = categories_consider.union(set(meta_categories['freq_classes'][set_name]))
            elif set_name=="all":
                categories_consider = set(meta_categories["thing_classes"])

        thing_classes = [cat for cat in metadata_default.thing_classes if cat in categories_consider]
        cat2idx = {cat:idx for idx, cat in enumerate(thing_classes)}
        metadata_default.set(cat2idx=cat2idx)
        if not hasattr(metadata_default, "thing_dataset_id_to_contiguous_id"):
            id_map = {v['id']: i for i, v in enumerate(meta_categories['dict_classes'].values())}
            metadata_default.set(thing_dataset_id_to_contiguous_id=id_map)
        cat2id_orig = {metadata_default.thing_classes[val]:key for key, val in metadata_default.thing_dataset_id_to_contiguous_id.items()}
        thing_dataset_id_to_contiguous_id = {cat2id_orig[cat]:idx for idx, cat in enumerate(thing_classes)}
        metadata.set(obj_set=obj_set, thing_classes=thing_classes, thing_dataset_id_to_contiguous_id=thing_dataset_id_to_contiguous_id)
        select_cat = True
        update_args["categories_consider"] = categories_consider
        update_args["metadata_default"] = metadata_default
    else:
        thing_classes = metadata_default.thing_classes
        thing_dataset_id_to_contiguous_id = metadata_default.get("thing_dataset_id_to_contiguous_id", {i: i for i in range(len(thing_classes))})
        metadata.set(thing_classes=thing_classes, thing_dataset_id_to_contiguous_id=thing_dataset_id_to_contiguous_id)
        select_cat = False
    update_args["select_cat"] = select_cat

    # 4. Load box proposals 
    obj_prop = kwargs.get("obj_prop")
    if obj_prop is not None:
        logger.info("Loading object proposals, file {}.".format(obj_prop))   
        timer = Timer()
        obj_prop = PathManager.get_local_path(obj_prop)
        with contextlib.redirect_stdout(io.StringIO()):
            with open(obj_prop, 'rb') as fin:
                object_proposals = pickle.load(fin, encoding='latin1')
        dict_object_proposals = {int(sample[0]):sample[1] for sample in object_proposals}
        if timer.seconds() > 1:
            logger.info("Loading object proposals takes {:.2f} seconds.".format(timer.seconds()))
        metadata.set(obj_prop=obj_prop)
        add_prop = True
        update_args["dict_object_proposals"] = dict_object_proposals
    else:
        add_prop = False
    update_args["add_prop"] = add_prop
    
    # 5. Add extra annotations for every image 
    logger.info("Updating sample dicts for {}.".format(name)) 
    timer = Timer()
    new_data_dicts = []
    for img_dict in data_dicts:
        new_data_dicts.append(update_lvis_image_dict(img_dict, **update_args))
    if timer.seconds() > 1:
        logger.info("Updatings samples takes {:.2f} seconds in sequential form.".format(timer.seconds()))
    
    # 6. register a function which returns dicts
    DatasetCatalog.register(name, lambda d=name: new_data_dicts)

    # 7. Add embeddings
    # Add noun embeddings
    obj_emb = kwargs.get("obj_file")
    if obj_emb is not None:
        logger.info("Adding object embeddings for {}".format(name))
        noun_emb_file = obj_emb
        with open(noun_emb_file, 'r') as fin:
            noun_embeddings = json.load(fin)

        class_embeddings = {}
        emb_dim = len(noun_embeddings[list(noun_embeddings.keys())[0]])
        # Adding background class
        class_emb_mtx = np.zeros((len(metadata.thing_classes) + 1, emb_dim), 
            dtype=np.float32)

        for idx, noun in enumerate(metadata.thing_classes):
            class_embeddings[idx] = np.asarray(noun_embeddings[noun], dtype=np.float32)
            class_emb_mtx[idx, :] = class_embeddings[idx]
        metadata.set(class_emb_mtx=class_emb_mtx)

def register_dataset(dataset_name):
    if dataset_name not in LVIS_DATASETS.keys():
        raise NotImplementedError('Not paths for dataset ' + dataset_name)

    dataset_paths = copy.deepcopy(LVIS_DATASETS["common_dict"])
    common_dict = 'common_val_dict' if '_val' in dataset_name else 'common_train_dict'
    dataset_paths.update(LVIS_DATASETS[common_dict])
    dataset_paths.update(LVIS_DATASETS[dataset_name])
    if 'caption' not in dataset_name:
        _ = dataset_paths.pop("cap_file")

    if dataset_name not in DatasetCatalog and dataset_name not in MetadataCatalog:
        extra_annotation_keys = []
        root_dir = dataset_paths.pop("img_dir")
        ann_file = dataset_paths.pop("ann_file")
        register_lvis_instances(dataset_name, ann_file, root_dir, 
            extra_annotation_keys=extra_annotation_keys, **dataset_paths)

    # Need to initialized to set thing_classes in dataset metadata
    dataset = DatasetCatalog.get(dataset_name)
    dataset_metadata = MetadataCatalog.get(dataset_name)


if __name__ == "__main__":
    """
    Test the LVIS instances dataset loader.

    Usage:
        python -m ovssd.data.datasets.lvis_instances coco_lvis_v1_caption_train_proposals
    """
    from detectron2.utils.logger import setup_logger
    from ovssd.utils.visualizer import Visualizer
    import sys
    from PIL import Image
    import numpy as np
    
    dataset_name = sys.argv[1]
    register_dataset(dataset_name)

    logger = setup_logger(name=__name__)
    assert dataset_name in DatasetCatalog.list()
    assert dataset_name in MetadataCatalog.list()
    meta = MetadataCatalog.get(dataset_name)
    dicts = DatasetCatalog.get(dataset_name)

    logger.info("Done loading {} samples.".format(len(dicts)))

    dirname = "lvis-data-vis"
    os.makedirs(dirname, exist_ok=True)
    num_imgs_to_vis = 20
    for i, d in enumerate(dicts):
        logger.info("Plotting sample {}, name {}.".format(i, d["file_name"]))
        img = np.array(Image.open(d["file_name"]))
        
        if "annotations" in d.keys():
            visualizer = Visualizer(img, metadata=meta)
            vis = visualizer.draw_dataset_dict(d)
            fpath = os.path.join(dirname, os.path.basename(d["file_name"]))
            vis.save(fpath)
        if "caption" in d.keys():
            visualizer = Visualizer(img, metadata=meta)
            vis = visualizer.draw_caption_dataset_dict(d)
            fpath = os.path.join(dirname, 'cap_'+os.path.basename(d["file_name"]))
            vis.save(fpath)
        if "proposal_boxes" in d.keys():
            visualizer = Visualizer(img, metadata=meta)
            vis = visualizer.draw_proposals_dataset_dict(d)
            fpath = os.path.join(dirname, 'prop_'+os.path.basename(d["file_name"]))
            vis.save(fpath)
        if i + 1 >= num_imgs_to_vis:
            break
    import ipdb; ipdb.set_trace()
