''' data '''
import random
from copy import deepcopy
from functools import partial

import numpy as np
import torch
import torch.utils.data
from core.data.prefetcher import PrefetchDataLoader
from core.utils.registry import Registry
from core.utils import logging
from core.utils.dist import get_dist_info
from .batch_sampler import build_batch_sampler
from .sr import PairedImageDataset

__all__ = [
    'build_dataset',
    'build_dataloader',
    'build_batch_sampler',
]

DATASET_REGISTRY = Registry('dataset')
DATASET_REGISTRY.register(PairedImageDataset)


def build_dataset(dataset_opt):
    """ Build dataset from options. """
    dataset_opt = deepcopy(dataset_opt)
    dataset = DATASET_REGISTRY.get(dataset_opt['type'])(dataset_opt)
    logging.info(
        f'Dataset [{dataset.__class__.__name__}] - {dataset_opt["name"]} '
        'is built.')
    return dataset


def build_dataloader(dataset,
                     dataset_opt,
                     num_gpu=1,
                     dist=False,
                     sampler=None,
                     batch_sampler=None,
                     seed=None):
    """ Build dataloader. """
    phase = dataset_opt['phase']
    if phase == 'train':
        if dist:  # distributed training
            batch_size = dataset_opt['batch_size_per_gpu']
            num_workers = dataset_opt['num_worker_per_gpu']
        else:  # non-distributed training
            multiplier = 1 if num_gpu == 0 else num_gpu
            batch_size = dataset_opt['batch_size_per_gpu'] * multiplier
            num_workers = dataset_opt['num_worker_per_gpu'] * multiplier
        dataloader_args = dict(dataset=dataset,
                               batch_size=batch_size,
                               shuffle=False,
                               num_workers=num_workers,
                               sampler=sampler,
                               batch_sampler=batch_sampler,
                               drop_last=True)
        if sampler is None:
            dataloader_args['shuffle'] = True
        if batch_sampler is not None:
            dataloader_args['sampler'] = None
            dataloader_args['batch_size'] = 1
            dataloader_args['shuffle'] = False
            dataloader_args['drop_last'] = False
        dataloader_args['worker_init_fn'] = partial(
            worker_init_fn,
            num_workers=num_workers,
            rank=get_dist_info()[0],
            seed=seed) if seed is not None else None
    elif phase in ['val', 'test']:  # validation
        dataloader_args = dict(dataset=dataset,
                               batch_size=1,
                               shuffle=False,
                               num_workers=0)
    else:
        raise ValueError(f'Wrong dataset phase: {phase}. '
                         "Supported ones are 'train', 'val' and 'test'.")

    dataloader_args['pin_memory'] = dataset_opt.get('pin_memory', False)
    dataloader_args['persistent_workers'] = dataset_opt.get(
        'persistent_workers', False)

    prefetch_mode = dataset_opt.get('prefetch_mode')
    if prefetch_mode == 'cpu':  # CPUPrefetcher
        num_prefetch_queue = dataset_opt.get('num_prefetch_queue', 1)
        logging.info(
            f'Use {prefetch_mode} prefetch dataloader: num_prefetch_queue = {num_prefetch_queue}'
        )
        return PrefetchDataLoader(num_prefetch_queue=num_prefetch_queue,
                                  **dataloader_args)
    # prefetch_mode=None: Normal dataloader
    # prefetch_mode='cuda': dataloader for CUDAPrefetcher
    return torch.utils.data.DataLoader(**dataloader_args)


def worker_init_fn(worker_id, num_workers, rank, seed):
    ''' Set the worker seed to num_workers * rank + worker_id + seed '''
    worker_seed = num_workers * rank + worker_id + seed
    np.random.seed(worker_seed)
    random.seed(worker_seed)
