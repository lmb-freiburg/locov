from detectron2.config import CfgNode

from ovr.data.mappers.basic_mappers import (
    BasicTextImageDatasetMapper,
    TextImageDatasetMapperNoise,
)
from ovr.data.mappers.coco_mappers import CocoImageDatasetMapper
from ovr.data.mappers.vaw_mappers import VawImageDatasetMapper


def get_mapper(dataset_name: str, cfg: CfgNode, is_train: bool):
    if "coco" in dataset_name:
        from detectron2.data import MetadataCatalog

        metadata = MetadataCatalog.get(dataset_name)
        mapper = CocoImageDatasetMapper(cfg, metadata, is_train)
        return mapper
    elif "vaw" in dataset_name:
        from detectron2.data import MetadataCatalog

        metadata = MetadataCatalog.get(dataset_name)
        mapper = VawImageDatasetMapper(cfg, metadata, is_train)
        return mapper
    elif "lvis" in dataset_name:
        from detectron2.data import MetadataCatalog

        metadata = MetadataCatalog.get(dataset_name)
        mapper = BasicTextImageDatasetMapper(cfg, is_train)
        return mapper
    else:
        from detectron2.data import MetadataCatalog

        metadata = MetadataCatalog.get(dataset_name)
        mapper = TextImageDatasetMapperNoise(cfg, metadata, is_train)
        return mapper
