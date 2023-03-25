''' criterions '''
from .psnr import calculate_psnr
from .ssim import calculate_ssim
from .niqe import calculate_niqe


__all__ = [
    'calculate_psnr',
    'calculate_ssim',
    'calculate_niqe',
]
