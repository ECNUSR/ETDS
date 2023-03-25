''' calculate_ssim '''
import numpy as np
import cv2
from core.data.transforms import reorder_image, bgr2ycbcr


def _ssim(img, img2):
    ''' ssim '''
    c1, c2 = (0.01 * 255)**2, (0.03 * 255)**2
    img = img.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    mu1 = cv2.filter2D(img, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq, mu2_sq = mu1**2, mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img * img2, -1, window)[5:-5, 5:-5] - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
    return ssim_map.mean()


def calculate_ssim(img, img2, crop_border, input_order='HWC', test_y_channel=False, **_kwargs):
    ''' calculate_ssim '''
    assert img.shape == img2.shape, (f'Image shapes are different: {img.shape}, {img2.shape}.')
    if input_order not in ['HWC', 'CHW']:
        raise ValueError(f'Wrong input_order {input_order}. Supported input_orders are ' '"HWC" and "CHW"')
    if not isinstance(crop_border, (list, tuple)):
        crop_border = (crop_border, crop_border)
    img = reorder_image(img, input_order=input_order)
    img2 = reorder_image(img2, input_order=input_order)
    img = img.astype(np.float64)
    img2 = img2.astype(np.float64)
    if crop_border[0] != 0:
        img = img[crop_border[0]:-crop_border[0], ...]
        img2 = img2[crop_border[0]:-crop_border[0], ...]
    if crop_border[1] != 0:
        img = img[:, crop_border[1]:-crop_border[1], ...]
        img2 = img2[:, crop_border[1]:-crop_border[1], ...]
    if test_y_channel:
        img = bgr2ycbcr(img.astype(np.float32) / 255., y_only=True)[..., None] * 255
        img2 = bgr2ycbcr(img2.astype(np.float32) / 255., y_only=True)[..., None] * 255
    ssims = []
    for i in range(img.shape[2]):
        ssims.append(_ssim(img[..., i], img2[..., i]))
    return np.array(ssims).mean()
