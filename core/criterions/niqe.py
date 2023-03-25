''' calculate_niqe '''
import os
import math
import numpy as np
import cv2
from scipy.ndimage import convolve
from scipy.special import gamma
from core.data.transforms import reorder_image, bgr2ycbcr, imresize


def estimate_aggd_param(block):
    ''' estimate_aggd_param '''
    block = block.flatten()
    gam = np.arange(0.2, 10.001, 0.001)  # len = 9801
    gam_reciprocal = np.reciprocal(gam)
    r_gam = np.square(gamma(gam_reciprocal * 2)) / (gamma(gam_reciprocal) * gamma(gam_reciprocal * 3))

    left_std = np.sqrt(np.mean(block[block < 0]**2))
    right_std = np.sqrt(np.mean(block[block > 0]**2))
    gammahat = left_std / right_std
    rhat = (np.mean(np.abs(block)))**2 / np.mean(block**2)
    rhatnorm = (rhat * (gammahat**3 + 1) * (gammahat + 1)) / ((gammahat**2 + 1)**2)
    array_position = np.argmin((r_gam - rhatnorm)**2)

    alpha = gam[array_position]
    beta_l = left_std * np.sqrt(gamma(1 / alpha) / gamma(3 / alpha))
    beta_r = right_std * np.sqrt(gamma(1 / alpha) / gamma(3 / alpha))
    return (alpha, beta_l, beta_r)


def compute_feature(block):
    ''' compute_feature '''
    feat = []
    alpha, beta_l, beta_r = estimate_aggd_param(block)
    feat.extend([alpha, (beta_l + beta_r) / 2])

    shifts = [[0, 1], [1, 0], [1, 1], [1, -1]]
    for shift in shifts:
        shifted_block = np.roll(block, shift, axis=(0, 1))
        alpha, beta_l, beta_r = estimate_aggd_param(block * shifted_block)
        # Eq. 8
        mean = (beta_r - beta_l) * (gamma(2 / alpha) / gamma(1 / alpha))
        feat.extend([alpha, mean, beta_l, beta_r])
    return feat


def niqe(img, mu_pris_param, cov_pris_param, gaussian_window, block_size_h=96, block_size_w=96):
    ''' niqe '''
    assert img.ndim == 2, ('Input image must be a gray or Y (of YCbCr) image with shape (h, w).')
    # crop image
    h, w = img.shape
    num_block_h = math.floor(h / block_size_h)
    num_block_w = math.floor(w / block_size_w)
    img = img[0:num_block_h * block_size_h, 0:num_block_w * block_size_w]

    distparam = []  # dist param is actually the multiscale features
    for scale in (1, 2):  # perform on two scales (1, 2)
        mu = convolve(img, gaussian_window, mode='nearest')
        sigma = np.sqrt(np.abs(convolve(np.square(img), gaussian_window, mode='nearest') - np.square(mu)))
        # normalize, as in Eq. 1 in the paper
        img_nomalized = (img - mu) / (sigma + 1)

        feat = []
        for idx_w in range(num_block_w):
            for idx_h in range(num_block_h):
                # process ecah block
                block = img_nomalized[idx_h * block_size_h // scale:(idx_h + 1) * block_size_h // scale,
                                      idx_w * block_size_w // scale:(idx_w + 1) * block_size_w // scale]
                feat.append(compute_feature(block))
        distparam.append(np.array(feat))
        if scale == 1:
            # img = imresize(img / 255., scale=0.5, antialiasing=True)
            img = imresize(img / 255., scalar_scale=0.5)
            img = img * 255.
    distparam = np.concatenate(distparam, axis=1)
    mu_distparam = np.nanmean(distparam, axis=0)
    distparam_no_nan = distparam[~np.isnan(distparam).any(axis=1)]
    cov_distparam = np.cov(distparam_no_nan, rowvar=False)
    invcov_param = np.linalg.pinv((cov_pris_param + cov_distparam) / 2)
    quality = np.matmul(
        np.matmul((mu_pris_param - mu_distparam), invcov_param), np.transpose((mu_pris_param - mu_distparam)))
    quality = np.sqrt(quality)
    quality = float(np.squeeze(quality))
    return quality


def calculate_niqe(img, crop_border, input_order='HWC', convert_to='y', **_kwargs):
    ''' calculate_niqe '''
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    niqe_pris_params = np.load(os.path.join(ROOT_DIR, 'niqe_pris_params.npz'))
    mu_pris_param = niqe_pris_params['mu_pris_param']
    cov_pris_param = niqe_pris_params['cov_pris_param']
    gaussian_window = niqe_pris_params['gaussian_window']
    img = img.astype(np.float32)
    if input_order != 'HW':
        img = reorder_image(img, input_order=input_order)
        if convert_to == 'y':
            img = bgr2ycbcr(img.astype(np.float32) / 255., y_only=True) * 255
        elif convert_to == 'gray':
            img = cv2.cvtColor(img / 255., cv2.COLOR_BGR2GRAY) * 255.
        img = np.squeeze(img)
    if crop_border != 0:
        img = img[crop_border:-crop_border, crop_border:-crop_border]
    img = img.round()
    return niqe(img, mu_pris_param, cov_pris_param, gaussian_window)
