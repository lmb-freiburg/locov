import torch
import os

from detectron2.data import DatasetMapper
from detectron2.data.dataset_mapper import DatasetMapper
from detectron2.data.common import DatasetFromList, MapDataset
from detectron2.data import samplers
from detectron2.data.build import get_detection_dataset_dicts, trivial_batch_collator
from detectron2.utils.comm import get_world_size

def build_detection_test_loader(cfg, dataset_name, mapper=None, output_dir=None):
    """
    Influenced from detectron2's own build_detection_test_loader but used to process mini-batch size greater than 1

    Args:
        cfg: a detectron2 CfgNode
        dataset_name (str): a name of the dataset that's available in the DatasetCatalog
        mapper (callable): a callable which takes a sample (dict) from dataset
           and returns the format to be consumed by the model.
           By default it will be `DatasetMapper(cfg, False)`.

    Returns:
        DataLoader: a torch DataLoader, that loads the given detection
        dataset, with test-time transformation and batching.
    """
    dataset_dicts = get_detection_dataset_dicts(
        [dataset_name],
        filter_empty=False,
        proposal_files=[
            cfg.DATASETS.PROPOSAL_FILES_TEST[
                list(cfg.DATASETS.TEST).index(dataset_name)
            ]
        ]
        if cfg.MODEL.LOAD_PROPOSALS
        else None,
    )

    if output_dir is not None:
        output_list = os.listdir(output_dir)
        output_set = set([json_file.split('.')[0] for json_file in output_list])
        new_dataset_dicts = []
        for video in dataset_dicts:
            video_name = video['file_name'].split('/')[-2]
            if video_name not in output_set:
                new_dataset_dicts.append(video)
        dataset_dicts = new_dataset_dicts

    dataset = DatasetFromList(dataset_dicts)
    if mapper is None:
        mapper = DatasetMapper(cfg, False)
    dataset = MapDataset(dataset, mapper)

    sampler = samplers.InferenceSampler(len(dataset))
    batch_sampler = torch.utils.data.sampler.BatchSampler(
        sampler, cfg.TEST.IMS_PER_BATCH, drop_last=False
    )

    data_loader = torch.utils.data.DataLoader(
        dataset,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        batch_sampler=batch_sampler,
        collate_fn=trivial_batch_collator,
    )
    return data_loader

def build_detection_val_loader(cfg, dataset_name, mapper=None, output_dir=None):
    """
    Influenced from detectron2's own build_detection_test_loader but used to process mini-batch size greater than 1

    Args:
        cfg: a detectron2 CfgNode
        dataset_name (str): a name of the dataset that's available in the DatasetCatalog
        mapper (callable): a callable which takes a sample (dict) from dataset
           and returns the format to be consumed by the model.
           By default it will be `DatasetMapper(cfg, False)`.

    Returns:
        DataLoader: a torch DataLoader, that loads the given detection
        dataset, with test-time transformation and batching.
    """
    dataset_dicts = get_detection_dataset_dicts(
        [dataset_name],
        filter_empty=False,
        proposal_files=[
            cfg.DATASETS.PROPOSAL_FILES_TEST[
                list(cfg.DATASETS.TEST).index(dataset_name)
            ]
        ]
        if cfg.MODEL.LOAD_PROPOSALS
        else None,
    )

    if output_dir is not None:
        output_list = os.listdir(output_dir)
        output_set = set([json_file.split('.')[0] for json_file in output_list])
        new_dataset_dicts = []
        for video in dataset_dicts:
            video_name = video['file_name'].split('/')[-2]
            if video_name not in output_set:
                new_dataset_dicts.append(video)
        dataset_dicts = new_dataset_dicts

    dataset = DatasetFromList(dataset_dicts)
    if mapper is None:
        mapper = DatasetMapper(cfg, False)
    dataset = MapDataset(dataset, mapper)

    sampler = samplers.InferenceSampler(len(dataset))
    # sampler = samplers.TrainingSampler(len(dataset))
    batch_size = max(cfg.SOLVER.IMS_PER_BATCH // get_world_size() - 1, 1)
    batch_sampler = torch.utils.data.sampler.BatchSampler(
        sampler, batch_size, drop_last=True
    )

    data_loader = torch.utils.data.DataLoader(
        dataset,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        batch_sampler=batch_sampler,
        collate_fn=trivial_batch_collator,
    )
    return data_loader