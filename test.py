''' test '''
from os import path as osp
import torch
from core.data import build_dataloader, build_dataset
from core.models import build_model
from core.utils import make_exp_dirs, logging
from core.utils.options import dict2str, parse_options
from core.utils.misc import get_time_str


def test_pipeline():
    ''' test pipeline '''
    opt, _ = parse_options(is_train=False)

    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    # mkdir and initialize loggers
    make_exp_dirs(opt)
    log_file = osp.join(opt['path']['log'],
                        f"test_{opt['name']}_{get_time_str()}.log")
    logging.init_logger(log_level='INFO',
                        log_file=log_file)
    logging.info(dict2str(opt))

    # create test dataset and dataloader
    test_loaders = []
    for _, dataset_opt in sorted(opt['datasets'].items()):
        test_set = build_dataset(dataset_opt)
        test_loader = build_dataloader(test_set,
                                       dataset_opt,
                                       num_gpu=opt['num_gpu'],
                                       dist=opt['dist'],
                                       sampler=None,
                                       seed=opt['manual_seed'])
        logging.info(
            f"Number of test images in {dataset_opt['name']}: {len(test_set)}")
        test_loaders.append(test_loader)

    # create model
    model = build_model(opt)

    for test_loader in test_loaders:
        test_set_name = test_loader.dataset.opt['name']
        logging.info(f'Testing {test_set_name}...')
        model.validation(test_loader, current_iter=opt['name'], **opt['val'])


if __name__ == '__main__':
    test_pipeline()
