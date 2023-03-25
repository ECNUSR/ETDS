''' Bicubic '''
from torch import nn
from torch.nn import functional as F


class Bicubic(nn.Module):
    ''' Bicubic '''
    def __init__(self, upscale):
        super().__init__()
        self.upscale = upscale

    def forward(self, x):
        ''' forward '''
        return F.interpolate(x, scale_factor=self.upscale, mode='bicubic')
