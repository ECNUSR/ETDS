''' models '''
from copy import deepcopy
from core.utils import logging
from core.utils.registry import Registry
from .ir import IRModel, ETDSModel

__all__ = ['build_model']

MODEL_REGISTRY = Registry('model')
MODEL_REGISTRY.register(IRModel)
MODEL_REGISTRY.register(ETDSModel)


def build_model(opt):
    """ Build model from options. """
    opt = deepcopy(opt)
    model = MODEL_REGISTRY.get(opt['model_type'])(opt)
    logging.info(f'Model [{model.__class__.__name__}] is created.')
    return model
