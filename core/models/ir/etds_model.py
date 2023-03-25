''' ir model '''
from collections import OrderedDict
import torch
from .ir_model import IRModel


class ETDSModel(IRModel):
    """ ETDS IR model for single image super-resolution. """
    def __init__(self, opt):
        self.fixed_residual_model_iters = opt['train']['fixed_residual_model_iters']        # number of rounds with K_r and K_{r2b} fixed parameters
        self.interpolation_loss_weight = opt['train']['interpolation_loss_weight']          # \alpha in Eq. 15
        super().__init__(opt)

    def update_learning_rate(self, current_iter, warmup_iter=-1):
        ''' update learning rate '''
        if current_iter == 1:
            # Set K_r, K_{r2b}, and K_{b2r} not to be trained at the beginning
            for name, param in self.network_g.named_parameters():
                if 'residual' in name:
                    param.requires_grad_(False)
        elif current_iter == self.fixed_residual_model_iters:
            # Set K_r, and K_{r2b} to participate in training after {fixed_residual_model_iters} rounds
            for name, param in self.network_g.named_parameters():
                if 'residual' in name:
                    param.requires_grad_(True)
        super().update_learning_rate(current_iter, warmup_iter)

    def optimize_parameters(self):
        self.optimizer_g.zero_grad()
        self.output, output2 = self.network_g(self.lq)

        l_total = 0
        loss_dict = OrderedDict()
        for _, (names, criterion) in self.criterions.items():
            # \mathcal{L}_{DS} = \mathcal{L}_{HF} + \alpha * \mathcal{L}_{LF}
            # \alpha equal to {interpolation_loss_weight}
            losses = criterion(self.output, self.gt) + self.interpolation_loss_weight * criterion(output2, self.gt)
            if not isinstance(losses, (tuple, list)):
                losses = [losses]
            assert len(names) == len(
                losses), 'The lengths of names and losses should match.'
            for name, loss in zip(names, losses):
                loss_dict[name] = loss
                l_total += loss
        l_total.backward()
        self.optimizer_g.step()
        self.log_dict = self.reduce_loss_dict(loss_dict)
        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    def test(self):
        ''' test '''
        if hasattr(self, 'network_g_ema'):
            self.network_g_ema.eval()
            with torch.no_grad():
                self.output = self.network_g_ema(self.lq)[0]
        else:
            self.network_g.eval()
            with torch.no_grad():
                self.output = self.network_g(self.lq)[0]
            self.network_g.train()
