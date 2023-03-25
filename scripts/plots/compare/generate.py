''' compare '''
import os
import os.path as osp
import math
import numpy as np
import cv2
import imageio
from matplotlib import pyplot as plt


def calc_psnr(img1, img2):
    ''' calculate PSNR '''

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))


def ssim(img1, img2):
    ''' SSIM '''
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calc_ssim(img1, img2):
    ''' calculate SSIM '''
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    if img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for _ in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        if img1.shape[2] == 1:
            return ssim(img1.squeeze(), img2.squeeze())
    raise ValueError('Wrong input dims in calc_ssim')


def calc_metrics(img1, img2):
    ''' calculate metrics '''
    return calc_psnr(img1, img2), calc_ssim(img1.astype(np.uint8), img2.astype(np.uint8))


def get_direction(bottom, top, left, right, h, w, window, bigwindow):
    ''' get direction '''
    side = min(h, w, bigwindow)
    r = (side - window) // 2
    bottom_r, top_r = bottom, h - top
    left_r, right_r = left, w - right
    if bottom_r >= r and top_r >= r:
        bottom_r = top_r = r
    elif bottom_r >= r:
        bottom_r = side - window - top_r
    else:
        top_r = side - window - bottom_r
    if left_r >= r and right_r >= r:
        left_r = right_r = r
    elif left_r >= r:
        left_r = side - window - right_r
    else:
        right_r = side - window - left_r

    return bottom - bottom_r, top + top_r, left - left_r, right + right_r


def scandir(base_path, dir_list):
    ''' scandir '''
    for entry in os.scandir(base_path):
        if not entry.name.startswith('.') and entry.is_file():
            yield entry.path, [osp.join(name, entry.name) for name in dir_list]
        else:
            yield from scandir(entry.path, [osp.join(name, entry.name) for name in dir_list])


def generate_patch(GT_path, dir_lists, patch_config, nometrics):
    ''' generate patchs (last image have best PSNR) '''
    # load hr img
    GT_img = imageio.imread(GT_path, pilmode='RGB')
    dir_imgs = [imageio.imread(path, pilmode='RGB') for path in dir_lists]

    h, w = GT_img.shape[:-1]
    for y in range(0, h-patch_config['window'], patch_config['step']):
        for x in range(0, w-patch_config['window'], patch_config['step']):
            GT_patch = GT_img[y:y+patch_config['window'], x:x+patch_config['window'], :]
            dir_patchs = [img[y:y+patch_config['window'], x:x+patch_config['window'], :] for img in dir_imgs]
            metrics = [calc_metrics(GT_patch, patch) if not nometric else (0, 0) for patch, nometric in zip(dir_patchs, nometrics)]
            if max(m[0] for m in metrics) == metrics[-1][0] and max(m[1] for m in metrics) == metrics[-1][1]:
                GT_img_copy = GT_img.copy()
                cv2.rectangle(GT_img_copy, (x, y), (x+patch_config['window']-1, y+patch_config['window']-1), (255, 0, 0), 2)
                bottom, top, left, right = get_direction(y, y + patch_config['window'], x, x + patch_config['window'], h, w, patch_config['window'], patch_config['bigwindow'])
                yield GT_img_copy[bottom:top, left:right, :], dir_patchs, metrics


def set_ax_no_border(ax):
    ''' set ax no border '''
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)


def plot_compare(GT_patch, dir_patchs, metrics, figsize, bigsize, dir_lists, save_path, info):
    ''' plot compare '''
    # create figure
    unit = 10                       # 小图的尺寸
    blank = 3                       # 文本的尺寸 和 小图之间的差距
    side = 0.2                      # 一圈的白边

    lh = bigsize[0] * (unit + blank) - blank                # 大图的高
    lw = bigsize[1] * (unit + blank) - blank                # 大图的宽
    h = figsize[0] * (unit + blank) + 2 * side              # 总图的高
    w = figsize[1] * (unit + blank) - blank + 2 * side      # 总图的宽

    fig = plt.figure(figsize=(w, h))

    ax = fig.add_axes([side/w, (side + (figsize[0] - bigsize[0]) * (unit + blank) + blank)/h, lw/w, lh/h])
    set_ax_no_border(ax)
    ax.imshow(GT_patch)
    ax_txt = fig.add_axes([side/w, (side + (figsize[0] - bigsize[0]) * (unit + blank) + 0.3 * blank)/h, lw/w, 0.7*blank/h])
    set_ax_no_border(ax_txt)
    ax_txt.set_title(label=info, y=0, fontdict={'fontsize': 100, 'family': 'Times New Roman'})

    index = 0
    for hi in range(figsize[0]):
        for wi in range(figsize[1]):
            if hi < bigsize[0] and wi < bigsize[1]:
                continue
            ax = fig.add_axes([(side + wi * (unit + blank))/w, (side + (figsize[0] - hi - 1) * (unit + blank) + blank)/h, unit/w, unit/h])
            set_ax_no_border(ax)
            ax.imshow(dir_patchs[index])
            ax_txt = fig.add_axes([(side + wi * (unit + blank))/w, (side + (figsize[0] - hi - 1) * (unit + blank) + 0.3 * blank)/h, unit/w, 0.9*blank/h])
            set_ax_no_border(ax_txt)
            if metrics[index][0] == metrics[index][1] == 0:
                ax_txt.set_title(label=f'{dir_lists[index]}\nPSNR/SSIM', y=0, fontdict={'fontsize': 100, 'family': 'Times New Roman'})
            elif index == len(dir_lists) - 1:
                ax_txt.set_title(label=f'{dir_lists[index]}\n{metrics[index][0]:.2f}/{metrics[index][1]:.4f}', y=0, fontdict={'fontsize': 100, 'fontweight': 700, 'family': 'Times New Roman'})
            else:
                ax_txt.set_title(label=f'{dir_lists[index]}\n{metrics[index][0]:.2f}/{metrics[index][1]:.4f}', y=0, fontdict={'fontsize': 100, 'family': 'Times New Roman'})
            index += 1

    plt.savefig(f'{save_path}.jpg')
    plt.savefig(f'{save_path}.pdf')

    plt.close()


def main():
    ''' main '''
    # hyperparameters
    figsize = (4, 3)        # height=4, width=3
    # figsize = (2, 6)        # height=2, width=6
    bigsize = (2, 2)        # big_image'unit size is 2x2
    # if have 'Ground Truth', his label will be given special consideration, (Ours in last)
    dir_lists = ['Ground Truth', 'Bicubic', 'ESPCN', 'FSRCNN', 'Plain-M6C32', 'ECBSR-M6C40', 'ABPN-M6C40', 'ETDS-L']
    patch_config = {
        'step': 16,
        'window': 64,
        'bigwindow': 512,
    }

    # check legality
    assert figsize[0] * figsize[1] == bigsize[0] * bigsize[1] + len(dir_lists)

    # work (all images need to be saved under the visiuals folder)
    base_path = osp.join('scripts', 'plots', 'compare', 'visiuals')
    for GT_path, dir_paths in scandir(osp.join(base_path, 'Ground Truth'), [osp.join(base_path, name) for name in dir_lists]):
        save_base_path = osp.splitext(GT_path.replace(osp.join('visiuals', 'Ground Truth'), 'best_compares'))[0]
        os.makedirs(save_base_path, exist_ok=True)
        paths = osp.splitext(GT_path)[0].split(os.sep)
        info = f'{paths[-1]} from {paths[5]}'
        for index, (GT_patch, dir_patchs, metrics) in enumerate(generate_patch(GT_path, dir_paths, patch_config, nometrics=[name == 'Ground Truth' for name in dir_lists])):
            plot_compare(GT_patch, dir_patchs, metrics, figsize, bigsize, dir_lists, osp.join(save_base_path, f'{index}'), info)


if __name__ == '__main__':
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    main()
