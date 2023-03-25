''' archs '''
from copy import deepcopy
from os import path as osp

from core.utils import logging
from core.utils.registry import Registry

from .ir import Bicubic, Plain, ESPCN, FSRCNN, ECBSR, ECBSRT, ECBSRET, ABPN, ABPN_ET, ETDS, ETDSForInference, ETDSOnlyET, ETDSAblationResidual, ETDSAblationAdvancedResidual

__all__ = ['build_network', 'get_network_path']
ARCH_REGISTRY = Registry('arch')
ARCH_REGISTRY.register(Bicubic)
ARCH_REGISTRY.register(Plain)
ARCH_REGISTRY.register(ESPCN)
ARCH_REGISTRY.register(FSRCNN)
ARCH_REGISTRY.register(ECBSR)
ARCH_REGISTRY.register(ECBSRT)
ARCH_REGISTRY.register(ECBSRET)
ARCH_REGISTRY.register(ABPN)
ARCH_REGISTRY.register(ABPN_ET)
ARCH_REGISTRY.register(ETDS)
ARCH_REGISTRY.register(ETDSForInference)
ARCH_REGISTRY.register(ETDSOnlyET)
ARCH_REGISTRY.register(ETDSAblationResidual)
ARCH_REGISTRY.register(ETDSAblationAdvancedResidual)


def get_network_path(opt):
    ''' get_network_path '''
    return ARCH_REGISTRY.get(opt['type']).__module__.replace('.', osp.sep) + '.py'


def build_network(opt):
    ''' build network '''
    opt = deepcopy(opt)
    network_type = opt.pop('type')
    net = ARCH_REGISTRY.get(network_type)(**opt)
    logging.info(f'Network [{net.__class__.__name__}] is created.')
    return net
