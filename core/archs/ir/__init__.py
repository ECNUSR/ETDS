''' ir archs '''
from .Bicubic import Bicubic
from .Plain import Plain
from .ESPCN import ESPCN
from .FSRCNN import FSRCNN
from .ECBSR import ECBSR, ECBSRT, ECBSRET
from .ABPN import ABPN, ABPN_ET
from .ETDS import ETDS, ETDSForInference, ETDSOnlyET, ETDSAblationResidual, ETDSAblationAdvancedResidual


__all__ = [
    'Bicubic',
    'ESPCN',
    'FSRCNN',
    'ECBSR',
    'ECBSRT',
    'ECBSRET',
    'ABPN',
    'ABPN_ET',
    'ETDS',
    'ETDSForInference',
    'ETDSOnlyET',
    'ETDSAblationResidual',
    'ETDSAblationAdvancedResidual',
]
