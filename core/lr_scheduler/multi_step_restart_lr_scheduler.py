''' MultiStepRestartLR '''
from collections import Counter
from torch.optim.lr_scheduler import _LRScheduler


class MultiStepRestartLR(_LRScheduler):
    """ MultiStep with restarts learning rate scheme.

    Args:
        optimizer (torch.nn.optimizer): Torch optimizer.
        milestones (list): Iterations that will decrease learning rate.
        gamma (float): Decrease ratio. Default: 0.1.
        restarts (list): Restart iterations. Default: [0].
        restart_weights (list): Restart weights at each restart iteration.
            Default: [1].
        last_epoch (int): Used in _LRScheduler. Default: -1.
    """

    def __init__(self, optimizer, milestones, gamma=0.1, restarts=(0, ), restart_weights=(1, ), last_epoch=-1):
        self.milestones = Counter(milestones)
        self.gamma = gamma
        self.restarts = restarts
        self.restart_weights = restart_weights
        assert len(self.restarts) == len(self.restart_weights), 'restarts and their weights do not match.'
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch in self.restarts:
            weight = self.restart_weights[self.restarts.index(self.last_epoch)]
            return [group['initial_lr'] * weight for group in self.optimizer.param_groups]
        if self.last_epoch not in self.milestones:
            return [group['lr'] for group in self.optimizer.param_groups]
        return [group['lr'] * self.gamma**self.milestones[self.last_epoch] for group in self.optimizer.param_groups]
