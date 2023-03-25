''' crop '''
import random
import torch


def mod_crop(img, scale):
    """ Mod crop images, used during testing. """
    img = img.copy()
    if img.ndim in (2, 3):
        h, w = img.shape[0], img.shape[1]
        h_remainder, w_remainder = h % scale, w % scale
        img = img[:h - h_remainder, :w - w_remainder, ...]
    else:
        raise ValueError(f'Wrong img ndim: {img.ndim}.')
    return img


def border_crop(imgs, patch_size):
    """ Paired random crop. Support Numpy array and Tensor inputs. """
    if not isinstance(imgs, (list, tuple)):
        imgs = [imgs]
    if not isinstance(patch_size, (list, tuple)):
        patch_size = [patch_size, patch_size]

    if torch.is_tensor(imgs[0]):
        imgs = [v[:, :, 0:patch_size[0], 0:patch_size[1]] for v in imgs]
    else:
        imgs = [v[0:patch_size[0], 0:patch_size[1], ...] for v in imgs]
    if len(imgs) == 1:
        imgs = imgs[0]
    return imgs


def patch_crop(imgs, top, bottom, left, right):
    ''' patch crop [top, bottom, left, right] '''
    if not isinstance(imgs, (list, tuple)):
        imgs = [imgs]
    if torch.is_tensor(imgs[0]):
        imgs = [v[:, :, top:bottom, left:right] for v in imgs]
    else:
        imgs = [v[top:bottom, left:right, ...] for v in imgs]
    if len(imgs) == 1:
        imgs = imgs[0]
    return imgs


def random_crop(imgs, patch_size, return_status=False):
    """ random crop. Support Numpy array and Tensor inputs. """
    if not isinstance(imgs, (list, tuple)):
        imgs = [imgs]
    if not isinstance(patch_size, (list, tuple)):
        patch_size = [patch_size, patch_size]

    if torch.is_tensor(imgs[0]):
        h_gt, w_gt = imgs[0].size()[-2:]
    else:
        h_gt, w_gt = imgs[0].shape[0:2]
    top = random.randint(0, h_gt - patch_size[0])
    left = random.randint(0, w_gt - patch_size[1])

    imgs = patch_crop(imgs, top, top + patch_size[0], left, left + patch_size[1])
    if return_status:
        return imgs, (top, top + patch_size[0], left, left + patch_size[1])
    return imgs


def check_and_compute_pair_size(lq_patch_size, gt_patch_size, scale):
    ''' check_and_compute_pair_size '''
    def _to_list(x):
        if not isinstance(x, (list, tuple)):
            x = [x, x]
        assert len(x) == 2, 'patch len must equals to 2.'
        return x

    def _check_and_compute_pair_size(lqps, gtps, s):
        none_nums = (lqps is None) + (gtps is None) + (s is None)
        assert none_nums < 2, 'at least two values must be specified in lq_patch_size, gt_patch_size and scale.'
        if none_nums == 0:
            assert lqps * s == gtps, 'lq_patch_size * scale must equals to gt_patch_size.'
        elif lqps is None:
            assert gtps % s == 0, 'gt_patch_size must be divided by scale.'
            lqps = gtps // s
        elif gtps is None:
            gtps = lqps * s
        else:
            s = gtps / lqps
        assert lqps % 1 == 0 and gtps % 1 == 0 and s % 1 == 0, 'lq_patch_size and gt_patch_size and scale must be integers.'
        assert lqps <= gtps, 'lq_patch_size must be greater than gt_patch_size.'
        return lqps, gtps, s

    lq_patch_sizes, gt_patch_sizes, scales = _to_list(lq_patch_size), _to_list(gt_patch_size), _to_list(scale)
    lq_patch_sizes[0], gt_patch_sizes[0], scales[0] = _check_and_compute_pair_size(lq_patch_sizes[0], gt_patch_sizes[0], scales[0])
    lq_patch_sizes[1], gt_patch_sizes[1], scales[1] = _check_and_compute_pair_size(lq_patch_sizes[1], gt_patch_sizes[1], scales[1])
    return lq_patch_sizes, gt_patch_sizes, scales


def paired_random_crop(img_gts, img_lqs, lq_patch_size=None, gt_patch_size=None, scale=None):
    """ Paired random crop. Support Numpy array and Tensor inputs. """
    if not isinstance(img_gts, list):
        img_gts = [img_gts]
    if not isinstance(img_lqs, list):
        img_lqs = [img_lqs]

    lq_patch_size, gt_patch_size, scale = check_and_compute_pair_size(lq_patch_size, gt_patch_size, scale)
    if torch.is_tensor(img_gts[0]):
        (h_lq, w_lq), (h_gt, w_gt) = img_lqs[0].size()[-2:], img_gts[0].size()[-2:]
    else:
        (h_lq, w_lq), (h_gt, w_gt) = img_lqs[0].shape[0:2], img_gts[0].shape[0:2]
    assert h_lq * gt_patch_size[0] == h_gt * lq_patch_size[0] and w_lq * gt_patch_size[1] == w_gt * lq_patch_size[1], 'Scale mismatches.'

    # randomly choose top and left coordinates for lq patch
    top = random.randint(0, h_lq - lq_patch_size[0])
    left = random.randint(0, w_lq - lq_patch_size[1])
    img_lqs = patch_crop(img_lqs, top, top + lq_patch_size[0], left, left + lq_patch_size[1])
    top_gt, left_gt = int(top * scale[0]), int(left * scale[1])
    img_gts = patch_crop(img_gts, top_gt, top_gt + gt_patch_size[0], left_gt, left_gt + gt_patch_size[1])

    return img_gts, img_lqs
