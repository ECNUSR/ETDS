''' augment '''
import random
import cv2
import torch


def get_channel(img):
    ''' get_channel '''
    if isinstance(img, (list, tuple)):
        img = img[0]
    if torch.is_tensor(img): # BCHW
        return img.shape[1]
    return img.shape[2]      # HWC


def augment_single_img(img, hflip, vflip, rot90, channels=None):
    ''' augment_single_img '''
    if torch.is_tensor(img): # BCHW
        if hflip:
            img = torch.flip(img, [2])
        if vflip:
            img = torch.flip(img, [3])
        if rot90:
            img = img.permute(0, 1, 3, 2)
        if channels is not None:
            img = img[:, channels, :, :]
    else:   # HWC
        if hflip:
            img = cv2.flip(img, 0)
        if vflip:
            img = cv2.flip(img, 1)
        if rot90:
            img = img.transpose(1, 0, 2)
        if channels is not None:
            img = img[:, :, channels]
    return img


def augment_single_flow(flow, hflip, vflip, rot90):
    ''' augment_single_flow '''
    if torch.is_tensor(flow): # BCHW
        if hflip:
            flow = torch.flip(flow, [2])
            flow[:, 1, :, :] *= -1
        if vflip:
            flow = torch.flip(flow, [3])
            flow[:, 0, :, :] *= -1
        if rot90:
            flow = flow.permute(1, 0, 2)
            flow = flow[:, [1, 0], :, :]
    else:   # HWC
        if hflip:
            flow = cv2.flip(flow, 0)
            flow[:, :, 1] *= -1
        if vflip:
            flow = cv2.flip(flow, 1)
            flow[:, :, 0] *= -1
        if rot90:
            flow = flow.transpose(1, 0, 2)
            flow = flow[:, :, [1, 0]]
    return flow


def augment_imgs(imgs, hflip, vflip, rot90, channels=None):
    """ augment_imgs """
    if not isinstance(imgs, (list, tuple)):
        imgs = [imgs]
    imgs = [augment_single_img(img, hflip, vflip, rot90, channels) for img in imgs]
    if len(imgs) == 1:
        imgs = imgs[0]
    return imgs


def augment_flows(flows, hflip, vflip, rot90):
    ''' augment_flows '''
    if not isinstance(flows, (list, tuple)):
        flows = [flows]
    flows = [augment_single_flow(flow, hflip, vflip, rot90) for flow in flows]
    if len(flows) == 1:
        flows = flows[0]
    return flows


def augment_imgs_inv(imgs, hflip, vflip, rot90):
    """ augment_imgs_inv """
    if rot90:
        hflip, vflip = vflip, hflip
    return augment_imgs(imgs, hflip, vflip, rot90)


def augment_flows_inv(flows, hflip, vflip, rot90):
    ''' augment_flows_inv '''
    if rot90:
        hflip, vflip = vflip, hflip
    return augment_flows(flows, hflip, vflip, rot90)


def random_augment(imgs, hflip=True, vflip=True, rotation=True, channels_shuffle=True, flows=None, return_status=False):
    """ Augment: horizontal flips OR rotate (0, 90, 180, 270 degrees). """
    hflip = hflip and random.random() < 0.5
    vflip = vflip and random.random() < 0.5
    rot90 = rotation and random.random() < 0.5
    if channels_shuffle:
        channels = list(range(get_channel(imgs)))
        random.shuffle(channels)
    else:
        channels = None

    imgs = augment_imgs(imgs, hflip, vflip, rot90, channels)
    if flows is not None:
        flows = augment_flows(flows, hflip, vflip, rot90)
        if return_status:
            return imgs, flows, (hflip, vflip, rot90, channels)
        return imgs, flows
    if return_status:
        return imgs, (hflip, vflip, rot90, channels)
    return imgs


def img_rotate(img, angle, center=None, scale=1.0):
    """ Rotate image. """
    (h, w) = img.shape[:2]
    if center is None:
        center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, scale)
    rotated_img = cv2.warpAffine(img, matrix, (w, h))
    return rotated_img
