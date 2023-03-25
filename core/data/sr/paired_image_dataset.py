''' paired image dataset '''
from torch.utils import data
from torchvision.transforms.functional import normalize

from core.data.paths import (paired_paths_from_folder, paired_paths_from_lmdb,
                             paired_paths_from_meta_info_file)
from core.utils import imfrombytes, img2tensor
from core.data.transforms import bgr2ycbcr, paired_random_crop, random_augment
from core.utils.file_client import FileClient


class PairedImageDataset(data.Dataset):
    """ Paired image dataset for image restoration. """
    def __init__(self, opt):
        super().__init__()

        self.opt = opt
        self.scale = opt['scale']

        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None

        folders, keys = [opt['dataroot_lq'], opt['dataroot_gt']], ['lq', 'gt']
        filename_tmpl = opt['filename_tmpl'] if 'filename_tmpl' in opt else '{}'

        if self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = folders
            self.io_backend_opt['client_keys'] = keys
            self.paths = paired_paths_from_lmdb(folders, keys)
        elif 'meta_info_file' in opt and opt['meta_info_file'] is not None:
            self.paths = paired_paths_from_meta_info_file(
                folders, keys, opt['meta_info_file'], filename_tmpl)
        else:
            self.paths = paired_paths_from_folder(folders, keys, filename_tmpl)

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'),
                                          **self.io_backend_opt)

        # Load gt and lq images. Dimension order: HWC; channel order: BGR; image range: [0, 1], float32.
        gt_path = self.paths[index]['gt_path']
        img_gt = imfrombytes(self.file_client.get(gt_path, 'gt'), float32=True)
        lq_path = self.paths[index]['lq_path']
        img_lq = imfrombytes(self.file_client.get(lq_path, 'lq'), float32=True)

        # augmentation for training
        if self.opt['phase'] == 'train':
            lq_size, gt_size = self.opt.get('lq_size', None), self.opt.get(
                'gt_size', None)
            # random crop
            img_gt, img_lq = paired_random_crop(img_gt,
                                                img_lq,
                                                lq_patch_size=lq_size,
                                                gt_patch_size=gt_size,
                                                scale=self.scale)
            # flip, rotation
            img_gt, img_lq = random_augment([img_gt, img_lq],
                                            self.opt.get('use_hflip', True),
                                            self.opt.get('use_vflip', True),
                                            self.opt.get('use_rot', True),
                                            self.opt.get(
                                                'use_channels_shuffle', True))

        # color space transform
        if 'color' in self.opt and self.opt['color'] == 'y':
            img_gt = bgr2ycbcr(img_gt, y_only=True)[..., None]
            img_lq = bgr2ycbcr(img_lq, y_only=True)[..., None]

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt, img_lq = img2tensor([img_gt, img_lq],
                                    bgr2rgb=True,
                                    float32=True)
        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)

        return {
            'lq': img_lq,
            'gt': img_gt,
            'lq_path': lq_path,
            'gt_path': gt_path
        }

    def __len__(self):
        return len(self.paths)
