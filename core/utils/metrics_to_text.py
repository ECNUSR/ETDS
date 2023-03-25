''' logger '''
import datetime
import time

from core.utils.dist import master_only


class MetricsToText():
    """ Message logger for printing. """
    def __init__(self, opt, start_iter=1):
        self.exp_name = opt['name']
        self.start_iter = start_iter
        self.max_iters = opt['train']['total_iter']
        self.start_time = time.time()

    def reset_start_time(self):
        ''' reset_start_time '''
        self.start_time = time.time()

    @master_only
    def convert(self, log_vars):
        """Format logging message.

        Args:
            log_vars (dict): It contains the following keys:
                epoch (int): Epoch number.
                iter (int): Current iter.
                lrs (list): List for learning rates.

                time (float): Iter time.
                data_time (float): Data time for each iter.
        """
        epoch = log_vars.pop('epoch')
        current_iter = log_vars.pop('iter')
        lrs = log_vars.pop('lrs')
        message = (
            f'[{self.exp_name[:5]}..][epoch:{epoch:3d}, iter:{current_iter:8,d}, lr:('
        )
        for v in lrs:
            message += f'{v:.3e},'
        message += ')] '
        if 'time' in log_vars.keys():
            iter_time = log_vars.pop('time')
            data_time = log_vars.pop('data_time')
            total_time = time.time() - self.start_time
            time_sec_avg = total_time / (current_iter - self.start_iter + 1)
            eta_sec = time_sec_avg * (self.max_iters - current_iter - 1)
            eta_str = str(datetime.timedelta(seconds=int(eta_sec)))
            message += f'[eta: {eta_str}, time (data): {iter_time:.3f} ({data_time:.3f})] '
        for k, v in log_vars.items():
            message += f'{k}: {v:.4e} '
        return message
