''' losses '''
from copy import deepcopy
from core.utils import logging
from core.utils.registry import Registry
from .losses import (CharbonnierLoss, L1Loss, MSELoss, WeightedTVLoss, PSNRLoss)


__all__ = [
    'L1Loss',
    'MSELoss',
    'CharbonnierLoss',
    'WeightedTVLoss',
    'PSNRLoss',
]
''' losses '''

__all__ = ['build_loss']

LOSS_REGISTRY = Registry('loss')
LOSS_REGISTRY.register(L1Loss)
LOSS_REGISTRY.register(MSELoss)
LOSS_REGISTRY.register(CharbonnierLoss)
LOSS_REGISTRY.register(WeightedTVLoss)
LOSS_REGISTRY.register(PSNRLoss)


def build_loss(opt):
    """ Build loss from options. """
    opt = deepcopy(opt)
    loss_type = opt.pop('type')
    loss = LOSS_REGISTRY.get(loss_type)(**opt)
    logging.info(f'Loss [{loss.__class__.__name__}] is created.')
    return loss
