''' train '''
import datetime
import math
import time
from os import path as osp
import torch
from core.data import build_batch_sampler, build_dataloader, build_dataset
from core.data.prefetcher import CPUPrefetcher, CUDAPrefetcher
from core.data.sampler import EnlargedSampler
from core.models import build_model
from core.utils import MetricsToText, check_resume, make_exp_dirs, logging
from core.utils.options import copy_networks_file, copy_opt_file, dict2str, parse_options
from core.utils.misc import mkdir_and_rename, scandir, get_time_str
from core.utils.timer import AvgTimer


def create_train_val_dataloader(opt):
    ''' create train and val dataloaders '''
    train_loader, val_loaders = None, []
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            dataset_enlarge_ratio = dataset_opt.get('dataset_enlarge_ratio', 1)
            train_set = build_dataset(dataset_opt)
            train_sampler = EnlargedSampler(train_set, opt['world_size'],
                                            opt['rank'], dataset_enlarge_ratio)
            train_batch_sample = build_batch_sampler(
                train_sampler, opt.get('batch_sampler', None))
            train_loader = build_dataloader(train_set,
                                            dataset_opt,
                                            num_gpu=opt['num_gpu'],
                                            dist=opt['dist'],
                                            sampler=train_sampler,
                                            batch_sampler=train_batch_sample,
                                            seed=opt['manual_seed'])

            num_iter_per_epoch = math.ceil(
                len(train_set) * dataset_enlarge_ratio /
                (dataset_opt['batch_size_per_gpu'] * opt['world_size']))
            total_iters = int(opt['train']['total_iter'])
            total_epochs = math.ceil(total_iters / (num_iter_per_epoch))
            logging.info(
                'Training statistics:'
                f'\n\tNumber of train images: {len(train_set)}'
                f'\n\tDataset enlarge ratio: {dataset_enlarge_ratio}'
                f'\n\tBatch size per gpu: {dataset_opt["batch_size_per_gpu"]}'
                f'\n\tWorld size (gpu number): {opt["world_size"]}'
                f'\n\tRequire iter number per epoch: {num_iter_per_epoch}'
                f'\n\tTotal epochs: {total_epochs}; iters: {total_iters}.')
        elif phase.split('_')[0] == 'val':
            val_set = build_dataset(dataset_opt)
            val_loader = build_dataloader(val_set,
                                          dataset_opt,
                                          num_gpu=opt['num_gpu'],
                                          dist=opt['dist'],
                                          sampler=None,
                                          seed=opt['manual_seed'])
            logging.info(
                f'Number of val images/folders in {dataset_opt["name"]}: {len(val_set)}'
            )
            val_loaders.append(val_loader)
        else:
            raise ValueError(f'Dataset phase {phase} is not recognized.')

    return train_loader, train_sampler, val_loaders, total_epochs, total_iters


def load_resume_state(opt):
    ''' load resume state '''
    resume_state_path = None
    if opt['auto_resume']:
        state_path = osp.join('experiments', opt['name'], 'training_states')
        if osp.isdir(state_path):
            states = list(
                scandir(state_path,
                        suffix='state',
                        recursive=False,
                        full_path=False))
            if len(states) != 0:
                states = [float(v.split('.state')[0]) for v in states]
                resume_state_path = osp.join(state_path,
                                             f'{max(states):.0f}.state')
                opt['resume']['state_path'] = resume_state_path
    else:
        if opt['resume'].get('state_path'):
            resume_state_path = opt['resume']['state_path']

    if resume_state_path is None:
        resume_state = None
    else:
        device_id = torch.cuda.current_device()
        resume_state = torch.load(
            resume_state_path,
            map_location=lambda storage, loc: storage.cuda(device_id))
        check_resume(opt, resume_state['iter'])
    return resume_state


def train_pipeline():
    ''' train pipeline '''
    opt, args = parse_options(is_train=True)

    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    # load resume states if necessary
    resume_state = load_resume_state(opt)
    # mkdir for experiments
    if resume_state is None:
        make_exp_dirs(opt)
        if opt['logger'].get('use_tb_logger') and opt['rank'] == 0:
            mkdir_and_rename(osp.join('tb_logger', opt['name']))

    # copy the yml file to the experiment root
    copy_opt_file(args.opt, opt['path']['experiments_root'])
    copy_networks_file(opt)

    log_file = osp.join(opt['path']['log'],
                        f"train_{opt['name']}_{get_time_str()}.log")
    if opt['logger'].get('use_tb_logger') and opt['rank'] == 0:
        tb_log_dir = osp.join('tb_logger', opt['name'])
    else:
        tb_log_dir = None
    logging.init_logger(log_level='INFO',
                        log_file=log_file,
                        tb_log_dir=tb_log_dir)
    logging.info(dict2str(opt))

    # create train and validation dataloaders
    result = create_train_val_dataloader(opt)
    train_loader, train_sampler, val_loaders, total_epochs, total_iters = result

    # create model
    model = build_model(opt)
    if resume_state:  # resume training
        model.resume_training(resume_state)  # handle optimizers and schedulers
        logging.info(f"Resuming training from epoch: {resume_state['epoch']}, "
                     f"iter: {resume_state['iter']}.")
        start_epoch = resume_state['epoch']
        current_iter = resume_state['iter']
    else:
        start_epoch = 0
        current_iter = 0

    metrics_converter = MetricsToText(opt, current_iter)

    # dataloader prefetcher
    prefetch_mode = opt['datasets']['train'].get('prefetch_mode')
    if prefetch_mode is None or prefetch_mode == 'cpu':
        prefetcher = CPUPrefetcher(train_loader)
    elif prefetch_mode == 'cuda':
        prefetcher = CUDAPrefetcher(train_loader, opt)
        logging.info(f'Use {prefetch_mode} prefetch dataloader')
        if opt['datasets']['train'].get('pin_memory') is not True:
            raise ValueError('Please set pin_memory=True for CUDAPrefetcher.')
    else:
        raise ValueError(f'Wrong prefetch_mode {prefetch_mode}.'
                         "Supported ones are: None, 'cuda', 'cpu'.")

    # training
    logging.info(
        f'Start training from epoch: {start_epoch}, iter: {current_iter}')
    data_timer, iter_timer = AvgTimer(), AvgTimer()
    start_time = time.time()

    for epoch in range(start_epoch, total_epochs + 1):
        train_sampler.set_epoch(epoch)
        prefetcher.reset()
        train_data = prefetcher.next()

        while train_data is not None:
            data_timer.record()

            current_iter += 1
            if current_iter > total_iters:
                break
            # update learning rate
            model.update_learning_rate(current_iter,
                                       warmup_iter=opt['train'].get(
                                           'warmup_iter', -1))
            # training
            model.feed_data(train_data)
            model.optimize_parameters()
            iter_timer.record()
            if current_iter == 1:
                metrics_converter.reset_start_time()
            if current_iter % opt['logger']['print_freq'] == 0:
                log_vars = {
                    'epoch': epoch,
                    'iter': current_iter,
                    'lrs': model.get_current_learning_rate(),
                    'time': iter_timer.get_avg_time(),
                    'data_time': data_timer.get_avg_time(),
                }
                log_vars.update(model.get_current_log())
                logging.tb_log(current_iter, **log_vars)
                logging.info(metrics_converter.convert(log_vars))

            # save models and training states
            if current_iter % opt['logger']['save_checkpoint_freq'] == 0:
                logging.info('Saving models and training states.')
                model.save(epoch, current_iter)

            # validation
            if opt.get('val') is not None and (current_iter %
                                               opt['val']['val_freq'] == 0):
                if len(val_loaders) > 1:
                    logging.warning(
                        'Multiple validation datasets are *only* supported by IRModel.'
                    )
                for val_loader in val_loaders:
                    model.validation(val_loader, current_iter, **opt['val'])

            data_timer.start()
            iter_timer.start()
            train_data = prefetcher.next()

    consumed_time = str(
        datetime.timedelta(seconds=int(time.time() - start_time)))
    logging.info(f'End of training. Time consumed: {consumed_time}')
    logging.info('Save the latest model.')
    model.save(epoch=-1, current_iter=-1)  # -1 stands for the latest
    if opt.get('val') is not None:
        for val_loader in val_loaders:
            model.validation(val_loader, current_iter, **opt['val'])


if __name__ == '__main__':
    train_pipeline()
