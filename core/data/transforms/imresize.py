''' imresize '''
from math import ceil
import numpy as np


def derive_size_from_scale(img_shape, scale):
    ''' derive_size_from_scale '''
    output_shape = []
    for k in range(2):
        output_shape.append(int(ceil(scale[k] * img_shape[k])))
    return output_shape


def derive_scale_from_size(img_shape_in, img_shape_out):
    ''' derive_scale_from_size '''
    scale = []
    for k in range(2):
        scale.append(1.0 * img_shape_out[k] / img_shape_in[k])
    return scale


def triangle(x, dtype):
    ''' triangle '''
    x = np.array(x).astype(dtype)
    lessthanzero = np.logical_and((x>=-1),x<0)
    greaterthanzero = np.logical_and((x<=1),x>=0)
    return np.multiply((x+1),lessthanzero) + np.multiply((1-x),greaterthanzero)


def cubic(x):
    ''' cubic '''
    x = np.array(x)
    absx = np.absolute(x)
    absx2 = np.multiply(absx, absx)
    absx3 = np.multiply(absx2, absx)
    return np.multiply(1.5*absx3 - 2.5*absx2 + 1, absx <= 1) + np.multiply(-0.5*absx3 + 2.5*absx2 - 4*absx + 2, (absx > 1) & (absx <= 2))


def contributions(in_length, out_length, scale, kernel, k_width):
    ''' contributions '''
    if scale < 1:
        h = lambda x: scale * kernel(scale * x)
        kernel_width = 1.0 * k_width / scale
    else:
        h = kernel
        kernel_width = k_width
    x = np.arange(1, out_length+1)
    u = x / scale + 0.5 * (1 - 1 / scale)
    left = np.floor(u - kernel_width / 2)
    ind = np.expand_dims(left, axis=1) + np.arange(int(ceil(kernel_width)) + 2) - 1
    indices = ind.astype(np.int32)
    weights = h(np.expand_dims(u, axis=1) - indices - 1)
    weights = np.divide(weights, np.expand_dims(np.sum(weights, axis=1), axis=1))
    aux = np.concatenate((np.arange(in_length), np.arange(in_length - 1, -1, step=-1))).astype(np.int32)
    indices = aux[np.mod(indices, aux.size)]
    ind2store = np.nonzero(np.any(weights, axis=0))
    weights = weights[:, ind2store]
    indices = indices[:, ind2store]
    return weights, indices


def imresizevec(inimg, weights, indices, dim):
    ''' imresizevec '''
    wshape = weights.shape
    if dim == 0:
        weights = weights.reshape((wshape[0], wshape[2], 1, 1))
        outimg =  np.sum(weights*((inimg[indices].squeeze(axis=1))), axis=1)
    elif dim == 1:
        weights = weights.reshape((1, wshape[0], wshape[2], 1))
        outimg =  np.sum(weights*((inimg[:, indices].squeeze(axis=2))), axis=2)
    if inimg.dtype == np.uint8:
        outimg = np.clip(outimg, 0, 255)
        return np.around(outimg).astype(np.uint8)
    return outimg


def resize_along_dim(img, dim, weights, indices):
    ''' resize_along_dim '''
    return imresizevec(img, weights, indices, dim)


def imresize(img, scalar_scale=None, method='bicubic', output_shape=None, force_float=False):
    ''' imresize '''
    if method == 'bicubic':
        kernel = cubic
    elif method == 'bilinear':
        kernel = triangle
    else:
        raise ValueError('unidentified kernel method supplied')
    dtype = img.dtype
    if force_float and dtype == np.uint8:
        img = img.astype(np.float64)
    kernel_width = 4.0
    if scalar_scale is not None and output_shape is not None:
        raise ValueError('either scalar_scale OR output_shape should be defined')
    if scalar_scale is not None:
        scalar_scale = float(scalar_scale)
        scale = [scalar_scale, scalar_scale]
        output_size = derive_size_from_scale(img.shape, scale)
    elif output_shape is not None:
        scale = derive_scale_from_size(img.shape, output_shape)
        output_size = list(output_shape)
    else:
        raise ValueError('either scalar_scale OR output_shape should be defined')
    weights, indices = [], []
    for k in range(2):
        w, ind = contributions(img.shape[k], output_size[k], scale[k], kernel, kernel_width)
        weights.append(w)
        indices.append(ind)
    flag2D = False
    order = np.argsort(np.array(scale))
    img = img.copy()
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
        flag2D = True
    for k in range(2):
        dim = order[k]
        img = resize_along_dim(img, dim, weights[dim], indices[dim])
    if flag2D:
        img = np.squeeze(img, axis=2)
    img = img.copy()
    if force_float and dtype == np.uint8:
        img = np.clip(img, 0, 255)
        img = np.around(img).astype(np.uint8)
    return img
