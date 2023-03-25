''' generate meta info '''
# pylint: disable=wrong-import-position
import sys
from os import path as osp
from PIL import Image
sys.path.append(sys.path[0].replace('scripts/datasets/DIV2K', ''))
from core.utils.misc import scandir         # pylint: disable=import-error


def generate_meta_info_div2k():
    """ Generate meta info for DIV2K dataset. """
    gt_folder = 'datasets/DIV2K/train/HR/subs'
    meta_info_txt = 'core/data/meta_info/meta_info_DIV2K800sub_GT.txt'
    img_list = sorted(list(scandir(gt_folder)))
    with open(meta_info_txt, 'w') as f:
        for img_path in img_list:
            img = Image.open(osp.join(gt_folder, img_path))  # lazy load
            width, height = img.size
            mode = img.mode
            if mode == 'RGB':
                n_channel = 3
            elif mode == 'L':
                n_channel = 1
            else:
                raise ValueError(f'Unsupported mode {mode}.')
            info = f'{img_path} ({height},{width},{n_channel})'
            f.write(f'{info}\n')


if __name__ == '__main__':
    generate_meta_info_div2k()
