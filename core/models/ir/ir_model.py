''' ir model '''
from collections import OrderedDict
from os import path as osp
import torch
from tqdm import tqdm
from core.archs import build_network
from core.losses import build_loss
from core.metrics import calculate_metric
from core.utils import imwrite, tensor2img, logging
from core.data.transforms import augment_imgs, augment_imgs_inv
from ..base_model import BaseModel


class IRModel(BaseModel):
    """ Base IR model for single image super-resolution. """
    def __init__(self, opt):
        super().__init__(opt)

        # define network
        self.network_g = build_network(opt['network_g'])
        self.network_g = self.model_to_device(self.network_g)
        self.print_network(self.network_g)

        # load pretrained models
        load_path = self.opt['resume'].get('network_g_path', None)
        if load_path is not None:
            network_g_strict = self.opt['resume'].get('network_g_strict', True)
            self.load_network(self.network_g, load_path, network_g_strict)

        if self.is_train:
            self.init_training_settings()

    def init_training_settings(self):
        ''' init_training_settings '''
        self.network_g.train()
        train_opt = self.opt['train']
        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logging.info(
                f'Use Exponential Moving Average with decay: {self.ema_decay}')
            self.network_g_ema = build_network(self.opt['network_g']).to(
                self.device)
            load_path = self.opt['resume'].get('network_g_ema_path', None)
            if load_path is not None:
                network_g_strict = self.opt['resume'].get(
                    'network_g_ema_strict', True)
                self.load_network(self.network_g_ema, load_path,
                                  network_g_strict)
            else:
                self.model_ema(0)  # copy network_g weight
            self.network_g_ema.eval()

        # define losses
        self.criterions = {}
        for name, cri_opt in train_opt['losses'].items():
            keys = cri_opt.pop('names') if 'names' in cri_opt else [name]
            self.criterions[name] = (keys, build_loss(cri_opt).to(self.device))

        if len(self.criterions) == 0:
            raise ValueError('Both pixel and perceptual losses are None.')

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        ''' setup_optimizers '''
        train_opt = self.opt['train']
        optim_params = []
        for k, v in self.network_g.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                logging.warning(f'Params {k} will not be optimized.')

        optim_type = train_opt['optim_g'].pop('type')
        self.optimizer_g = self.get_optimizer(optim_type, optim_params,
                                              **train_opt['optim_g'])
        self.optimizers.append(self.optimizer_g)

    def feed_data(self, data):
        ''' feed_data '''
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)

    def model_ema(self, decay=0.999):
        ''' do model ema '''
        network_g = self.get_bare_model(self.network_g)
        network_g_params = dict(network_g.named_parameters())
        network_g_ema_params = dict(self.network_g_ema.named_parameters())
        for k, network_g_ema_param in network_g_ema_params.items():
            network_g_ema_param.data.mul_(decay).add_(network_g_params[k].data,
                                                      alpha=1 - decay)

    def optimize_parameters(self):
        self.optimizer_g.zero_grad()
        self.output = self.network_g(self.lq)

        l_total = 0
        loss_dict = OrderedDict()
        for _, (names, criterion) in self.criterions.items():
            losses = criterion(self.output, self.gt)
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
                self.output = self.network_g_ema(self.lq)
        else:
            self.network_g.eval()
            with torch.no_grad():
                self.output = self.network_g(self.lq)
            self.network_g.train()

    def self_ensemble_test(self):
        ''' self_ensemble_test '''
        original_lq = self.lq
        outputs = []
        for hflip in [False, True]:
            for vflip in [False, True]:
                for rot90 in [False, True]:
                    self.lq = augment_imgs(original_lq,
                                           hflip=hflip,
                                           vflip=vflip,
                                           rot90=rot90)
                    self.test()
                    output = augment_imgs_inv(self.output,
                                              hflip=hflip,
                                              vflip=vflip,
                                              rot90=rot90)
                    outputs.append(output)
        self.lq = original_lq
        self.output = torch.cat(outputs).mean(0, True)

    def dist_validation(self, dataloader, current_iter, save_img, **kwargs):
        ''' dist_validation '''
        if self.opt['rank'] == 0:
            self.nondist_validation(dataloader, current_iter, save_img,
                                    **kwargs)

    def nondist_validation(self,
                           dataloader,
                           current_iter,
                           save_img,
                           metrics=None,
                           suffix=None,
                           is_self_ensemble=False,
                           **_kwargs):
        ''' nondist_validation '''
        dataset_name = dataloader.dataset.opt['name']
        use_pbar = self.opt['val'].get('pbar', False)

        if metrics is not None:
            if not hasattr(self,
                           'metric_results'):  # only execute in the first run
                self.metric_results = {metric: 0 for metric in metrics.keys()}
            # initialize the best metric results for each dataset_name (supporting multiple validation datasets)
            self._initialize_best_metric_results(dataset_name)
        # zero self.metric_results
        if metrics is not None:
            self.metric_results = {metric: 0 for metric in self.metric_results}

        metric_data = {}
        if use_pbar:
            pbar = tqdm(total=len(dataloader), unit='image')

        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
            self.feed_data(val_data)
            if is_self_ensemble:
                self.self_ensemble_test()
            else:
                self.test()

            visuals = self.get_current_visuals()
            metric_data['img'] = tensor2img([visuals['result']])
            if 'gt' in visuals:
                gt_img = tensor2img([visuals['gt']])
                metric_data['img2'] = gt_img
                del self.gt

            # tentative for out of GPU memory
            del self.lq
            del self.output
            torch.cuda.empty_cache()

            if save_img:
                if self.opt['is_train']:
                    save_img_path = osp.join(self.opt['path']['visualization'],
                                             img_name,
                                             f'{img_name}_{current_iter}.png')
                else:
                    if suffix is not None:
                        save_img_path = osp.join(
                            self.opt['path']['visualization'], dataset_name,
                            f'{img_name}_{suffix}.png')
                    else:
                        save_img_path = osp.join(
                            self.opt['path']['visualization'], dataset_name,
                            f'{img_name}_{self.opt["name"]}.png')
                imwrite(metric_data['img'], save_img_path)

            if metrics is not None:
                # calculate metrics
                for name, opt_ in metrics.items():
                    self.metric_results[name] += calculate_metric(
                        metric_data, opt_)
            if use_pbar:
                pbar.update(1)
                pbar.set_description(f'Test {img_name}')
        if use_pbar:
            pbar.close()

        if metrics is not None:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= (idx + 1)  # pylint: disable=undefined-loop-variable
                # update the best metric result
                self._update_best_metric_result(dataset_name, metric,
                                                self.metric_results[metric],
                                                current_iter)

            self._log_validation_metric_values(current_iter, dataset_name)

    def _log_validation_metric_values(self, current_iter, dataset_name):
        log_str = f'Validation [epoch {current_iter}] {dataset_name}\n'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}'
            if hasattr(self, 'best_metric_results'):
                log_str += (
                    f'\tBest: {self.best_metric_results[dataset_name][metric]["val"]:.4f} @ '
                    f'{self.best_metric_results[dataset_name][metric]["iter"]} iter'
                )
            log_str += '\n'

        logging.info(log_str)
        for metric, value in self.metric_results.items():
            logging.tb_log(current_iter,
                           **{f'metrics/{dataset_name}/{metric}': value})

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq.detach().cpu()
        out_dict['result'] = self.output.detach().cpu()
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()
        return out_dict

    def save(self, epoch, current_iter):
        self.save_network(self.network_g, 'network_g', current_iter)
        if hasattr(self, 'network_g_ema'):
            self.save_network(self.network_g_ema, 'network_g_ema',
                              current_iter)
        self.save_training_state(epoch, current_iter)
