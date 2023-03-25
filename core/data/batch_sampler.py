''' sample '''
from copy import deepcopy

from core.utils import logging
from core.utils.registry import Registry
from core.data.sampler import BatchSampler

__all__ = ['build_batch_sampler']

SAMPLER_REGISTRY = Registry('sampler')
SAMPLER_REGISTRY.register(BatchSampler)


def build_batch_sampler(sampler, opt):
    """ Build batch sampler from options. """
    if opt is None:
        return None
    opt = deepcopy(opt)
    sampler = SAMPLER_REGISTRY.get(opt['type'])(opt)
    logging.info(
        f'Batch sampler [{sampler.__class__.__name__}] - {opt["name"]} '
        'is built.')
    return sampler
