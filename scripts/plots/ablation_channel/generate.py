''' Plot the effect of each channel on psnr after ablation. '''
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
from torch import nn
from torch.nn import functional as F


def bgr2ycbcr(img, y_only=False):
    ''' bgr space to ycbcr space '''
    if y_only:
        out_img = np.dot(img, [24.966, 128.553, 65.481]) + 16.0
    else:
        out_img = np.matmul(
            img, [[24.966, 112.0, -18.214], [128.553, -74.203, -93.786], [65.481, -37.797, 112.0]]) + [16, 128, 128]
    return out_img / 255


def calculate_psnr(img, img2, crop_border):
    ''' calculate_psnr '''
    if not isinstance(crop_border, (list, tuple)):
        crop_border = (crop_border, crop_border)
    if crop_border[0] != 0:
        img = img[crop_border[0]:-crop_border[0], ...]
        img2 = img2[crop_border[0]:-crop_border[0], ...]
    if crop_border[1] != 0:
        img = img[:, crop_border[1]:-crop_border[1], ...]
        img2 = img2[:, crop_border[1]:-crop_border[1], ...]
    mse = np.mean((img - img2)**2)
    if mse == 0:
        return float('inf')
    return 20. * np.log10(255. / np.sqrt(mse))


class Plain(nn.Module):
    ''' Plain '''
    def __init__(self, num_in_ch, num_out_ch, upscale, num_block, num_feat):
        super().__init__()

        self.conv_first = nn.Conv2d(num_in_ch, num_feat, kernel_size=3, padding=1)

        backbone_convs = []
        for _ in range(num_block):
            backbone_convs.append(nn.Conv2d(num_feat, num_feat, 3, padding=1))
        self.backbone_convs = nn.ModuleList(backbone_convs)

        self.conv_last = nn.Conv2d(num_feat, num_out_ch * (upscale**2), 3, padding=1)

        self.upsampler = nn.PixelShuffle(upscale)

    def forward(self, input, idx):
        ''' forward '''
        x = F.relu(self.conv_first(input))

        for backbone_conv in self.backbone_convs:
            x = F.relu(backbone_conv(x))

        ablation = None
        if idx >= 0:
            ablation = x[:, idx, :, :].clone()
            x[:, idx, :, :] = 0

        x = self.upsampler(self.conv_last(x))
        return x, ablation


class ETDSForInference(nn.Module):
    ''' ETDS for inference '''
    def __init__(self, num_in_ch, num_out_ch, upscale, num_block, num_feat, num_residual_feat):
        super().__init__()
        assert (num_residual_feat >= num_in_ch) and (num_out_ch == num_in_ch)
        self.upscale = upscale

        self.conv_first = nn.Conv2d(num_in_ch, num_feat, kernel_size=3, padding=1)

        backbone_convs = []
        for _ in range(num_block):
            backbone_convs.append(nn.Conv2d(num_feat, num_feat, kernel_size=3, padding=1))
        self.backbone_convs = nn.ModuleList(backbone_convs)

        if upscale == 3:
            self.conv_last = nn.Conv2d(num_feat, num_out_ch * (upscale**2) + 1, kernel_size=3, padding=1)
            self.conv_clip = nn.Conv2d(num_out_ch * (upscale**2) + 1, num_out_ch * (upscale**2) + 1, 1)
        else:
            self.conv_last = nn.Conv2d(num_feat, num_out_ch * (upscale**2), kernel_size=3, padding=1)
            self.conv_clip = nn.Conv2d(num_out_ch * (upscale**2), num_out_ch * (upscale**2), 1)

        self.upsampler = nn.PixelShuffle(upscale)

    def forward(self, input, idx):
        ''' forward '''
        x = F.relu(self.conv_first(input))

        for backbone_conv in self.backbone_convs:
            x = F.relu(backbone_conv(x))

        ablation = None
        if idx >= 0:
            ablation = x[:, idx, :, :].clone()
            x[:, idx, :, :] = 0

        x = F.relu(self.conv_last(x))
        if self.upscale == 3:
            x = F.relu(self.conv_clip(x))[:, :-1, :, :]
        else:
            x = F.relu(self.conv_clip(x))

        x = self.upsampler(x)
        return x, ablation


class ETDSAblationAdvancedResidual(nn.Module):
    ''' ETDSAblationAdvancedResidual '''
    def __init__(self, num_in_ch, num_out_ch, upscale, num_block, num_feat, num_residual_feat):
        super().__init__()
        assert (num_feat > num_residual_feat >= num_in_ch) and (num_out_ch == num_in_ch)

        num_feat = num_feat - num_residual_feat

        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, padding=1)
        self.conv_residual_first = nn.Conv2d(num_in_ch, num_residual_feat, kernel_size=3, padding=1)

        backbone_convs = []
        residual_convs = []
        add_residual_convs = []
        advanced_residual_convs = []
        for _ in range(num_block):
            backbone_convs.append(nn.Conv2d(num_feat, num_feat, 3, padding=1))
            residual_convs.append(nn.Conv2d(num_residual_feat, num_residual_feat, kernel_size=3, padding=1))
            add_residual_convs.append(nn.Conv2d(num_residual_feat, num_feat, kernel_size=3, padding=1))
            advanced_residual_convs.append(nn.Conv2d(num_feat, num_residual_feat, kernel_size=3, padding=1))
        self.backbone_convs = nn.ModuleList(backbone_convs)
        self.residual_convs = nn.ModuleList(residual_convs)
        self.add_residual_convs = nn.ModuleList(add_residual_convs)
        self.advanced_residual_convs = nn.ModuleList(advanced_residual_convs)

        self.conv_last = nn.Conv2d(num_feat, num_out_ch * (upscale**2), 3, padding=1)
        self.conv_residual_last = nn.Conv2d(num_residual_feat, num_out_ch * (upscale**2), kernel_size=3, padding=1)

        self.upsampler = nn.PixelShuffle(upscale)

        self.init_weights(num_in_ch, upscale, num_residual_feat)

    def init_weights(self, colors, upscale, num_residual_feat):
        ''' init weights '''
        self.conv_residual_first.weight.data.fill_(0)
        for i in range(colors):
            self.conv_residual_first.weight.data[i, i, 1, 1] = 1
        self.conv_residual_first.bias.data.fill_(0)
        for residual_conv in self.residual_convs:
            residual_conv.weight.data.fill_(0)
            for i in range(num_residual_feat):
                residual_conv.weight.data[i, i, 1, 1] = 1
            residual_conv.bias.data.fill_(0)
        for add_residual_conv in self.add_residual_convs:
            add_residual_conv.weight.data.fill_(0)
            add_residual_conv.bias.data.fill_(0)
        for advanced_residual_conv in self.advanced_residual_convs:
            advanced_residual_conv.weight.data.fill_(0)
            advanced_residual_conv.bias.data.fill_(0)
        self.conv_residual_last.weight.data.fill_(0)
        for i in range(colors):
            for j in range(upscale**2):
                self.conv_residual_last.weight.data[i * (upscale**2) + j, i, 1, 1] = 1
        self.conv_residual_last.bias.data.fill_(0)

    def forward(self, input, idx):
        ''' forward '''
        x = F.relu(self.conv_first(input))
        r = F.relu(self.conv_residual_first(input))

        for backbone_conv, residual_conv, add_residual_conv, advanced_residual_conv in zip(self.backbone_convs, self.residual_convs, self.add_residual_convs, self.advanced_residual_convs):
            x, r = F.relu(backbone_conv(x) + add_residual_conv(r)), F.relu(residual_conv(r) + advanced_residual_conv(x))

        if idx < 0:
            ablation = 0
        elif idx < 29:
            ablation = x[:, idx, :, :].clone()
            x[:, idx, :, :] = 0
        else:
            ablation = r[:, idx - 29, :, :].clone()
            r[:, idx - 29, :, :] = 0

        x = self.upsampler(self.conv_last(x))
        r = self.upsampler(self.conv_residual_last(r))
        return x + r, ablation


def main():
    ''' main '''
    # set font
    plt.rc('font',family='Times New Roman')

    # load input
    LR = torch.from_numpy(cv2.cvtColor(cv2.imread('datasets/Set5/LRbicx4/butterfly.png'), cv2.COLOR_BGR2RGB) / 255.0).float().permute(2, 0, 1).contiguous().unsqueeze(0)
    GT = cv2.cvtColor(cv2.imread('datasets/Set5/GTmod12/butterfly.png'), cv2.COLOR_BGR2RGB)

    # plain
    os.makedirs('scripts/plots/ablation_channel/Plains', exist_ok=True)
    model1 = Plain(3, 3, 4, 6, 32)
    model1.load_state_dict(torch.load('experiments/pretrained_models/Plain/Plain_M6C32_x4.pth'))
    plain_psnrs = []
    for idx in range(32):
        with torch.no_grad():
            SR, ablation = model1(LR, idx)
        SR = np.clip(SR.squeeze().permute(1, 2, 0).numpy() * 255, 0, 255)
        ablation = ablation.squeeze().numpy() * 255
        plain_psnrs.append(calculate_psnr(SR, GT, 4))
        cv2.imwrite(f'scripts/plots/ablation_channel/Plains/{idx}.png', ablation)

    # EDTS
    os.makedirs('scripts/plots/ablation_channel/ETDSs', exist_ok=True)
    model2 = ETDSForInference(3, 3, 4, 6, 32, 3)
    model2.load_state_dict(torch.load('experiments/pretrained_models/ETDS/ETDS_M6C32_x4.pth'))
    pure_psnrs = []
    for idx in range(32):
        with torch.no_grad():
            SR, ablation = model2(LR, idx)
        SR = np.clip(SR.squeeze().permute(1, 2, 0).numpy() * 255, 0, 255)
        ablation = ablation.squeeze().numpy() * 255
        pure_psnrs.append(calculate_psnr(SR, GT, 4))
        cv2.imwrite(f'scripts/plots/ablation_channel/ETDSs/{idx}.png', ablation)

    # advance
    os.makedirs('scripts/plots/ablation_channel/Advances', exist_ok=True)
    model3 = ETDSAblationAdvancedResidual(3, 3, 4, 6, 32, 3)
    model3.load_state_dict(torch.load('experiments/pretrained_models/ETDS/ablations/ETDS_M6C32_x4_advance_residual.pth'))
    advance_psnrs = []
    for idx in range(32):
        with torch.no_grad():
            SR, ablation = model3(LR, idx)
        SR = np.clip(SR.squeeze().permute(1, 2, 0).numpy() * 255, 0, 255)
        ablation = ablation.squeeze().numpy() * 255
        advance_psnrs.append(calculate_psnr(SR, GT, 4))
        cv2.imwrite(f'scripts/plots/ablation_channel/Advances/{idx}.png', ablation)


    # hyp
    fontsize = 30

    fig, ax = plt.subplots(figsize=(15, 5))

    ax.plot(plain_psnrs, linewidth=5, label='Plain-M6C32')
    ax.plot(pure_psnrs, linewidth=5, label='ETDS-M')
    ax.plot(advance_psnrs, linewidth=5, label='Variant Ⅴ')

    plt.ylabel('PSNR (dB)', fontsize=fontsize)

    for size in ax.get_xticklabels():  # Set fontsize for x-axis
        size.set_fontsize(f'{fontsize}')
    for size in ax.get_yticklabels():  # Set fontsize for y-axis
        size.set_fontsize(f'{fontsize}')

    ax.legend(fontsize=fontsize-5, ncol=3)
    plt.show()

    fig.savefig('scripts/plots/ablation_channel/ablation_channel.svg', format='svg', dpi=300)



if __name__ == '__main__':
    main()
