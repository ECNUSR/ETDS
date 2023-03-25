''' calculate_psnr '''
import numpy as np
from core.data.transforms import reorder_image, bgr2ycbcr


def calculate_psnr(img, img2, crop_border, input_order='HWC', test_y_channel=False, **_kwargs):
    ''' calculate_psnr '''
    assert img.shape == img2.shape, (f'Image shapes are different: {img.shape}, {img2.shape}.')
    if input_order not in ['HWC', 'CHW']:
        raise ValueError(f'Wrong input_order {input_order}. Supported input_orders are ' '"HWC" and "CHW"')
    if not isinstance(crop_border, (list, tuple)):
        crop_border = (crop_border, crop_border)
    img = reorder_image(img, input_order=input_order).astype(np.float64)
    img2 = reorder_image(img2, input_order=input_order).astype(np.float64)
    if crop_border[0] != 0:
        img = img[crop_border[0]:-crop_border[0], ...]
        img2 = img2[crop_border[0]:-crop_border[0], ...]
    if crop_border[1] != 0:
        img = img[:, crop_border[1]:-crop_border[1], ...]
        img2 = img2[:, crop_border[1]:-crop_border[1], ...]
    if test_y_channel:
        img = bgr2ycbcr(img.astype(np.float32) / 255., y_only=True) * 255
        img2 = bgr2ycbcr(img2.astype(np.float32) / 255., y_only=True) * 255
    mse = np.mean((img - img2)**2)
    if mse == 0:
        return float('inf')
    return 20. * np.log10(255. / np.sqrt(mse))
