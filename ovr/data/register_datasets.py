from detectron2.data import DatasetCatalog, MetadataCatalog


def check_register(dataset_name):
    assert dataset_name in DatasetCatalog
    assert dataset_name in MetadataCatalog
    return


def get_register_dataset(dataset_name):
    if "coco" in dataset_name:
        from .datasets.coco_instances import register_dataset

        return register_dataset
    elif "vaw" in dataset_name:
        from .datasets.vaw_instances import register_dataset

        return register_dataset
    elif "lvis" in dataset_name:
        from .datasets.lvis_instances import register_dataset

        return register_dataset
