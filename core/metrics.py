''' metrics '''
from copy import deepcopy

from core.utils.registry import Registry
from core.criterions import calculate_niqe, calculate_psnr, calculate_ssim

__all__ = [
    'calculate_metric',
    'calculate_psnr',
    'calculate_ssim',
    'calculate_niqe',
]

METRIC_REGISTRY = Registry('metric')
METRIC_REGISTRY.register(calculate_psnr)
METRIC_REGISTRY.register(calculate_ssim)
METRIC_REGISTRY.register(calculate_niqe)


def calculate_metric(data, opt):
    """ Calculate metric from data and options. """
    opt = deepcopy(opt)
    metric_type = opt.pop('type')
    metric = METRIC_REGISTRY.get(metric_type)(**data, **opt)
    return metric
