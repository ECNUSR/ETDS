''' loss utils '''
import functools
from torch.nn import functional as F


def reduce_loss(loss, reduction):
    """ reduce_loss """
    reduction_enum = F._Reduction.get_enum(reduction)   # pylint: disable=protected-access
    if reduction_enum == 0:
        return loss
    if reduction_enum == 1:
        return loss.mean()
    return loss.sum()


def weight_reduce_loss(loss, weight=None, reduction='mean'):
    ''' weight_reduce_loss '''
    if weight is not None:
        assert weight.dim() == loss.dim()
        assert weight.size(1) == 1 or weight.size(1) == loss.size(1)
        loss = loss * weight
    if weight is None or reduction == 'sum':
        loss = reduce_loss(loss, reduction)
    elif reduction == 'mean':
        if weight.size(1) > 1:
            weight = weight.sum()
        else:
            weight = weight.sum() * loss.size(1)
        loss = loss.sum() / weight
    return loss


def weighted_loss(loss_func):
    ''' @weighted_loss '''
    @functools.wraps(loss_func)
    def wrapper(pred, target, weight=None, reduction='mean', **kwargs):
        loss = loss_func(pred, target, **kwargs)
        loss = weight_reduce_loss(loss, weight, reduction)
        return loss
    return wrapper
