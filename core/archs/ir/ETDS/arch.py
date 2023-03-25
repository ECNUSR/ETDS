''' ECBSR arch '''
import torch
from torch import nn
from torch.nn import functional as F

class ETDS(nn.Module):
    ''' ETDS arch
    args:
        num_in_ch: the number of channels of the input image
        num_out_ch: the number of channels of the output image
        upscale: scale factor
        num_block: the number of layers of the model
        num_feat: the number of channels of the model (after ET)
        num_residual_feat: the number of channels of the residual branch
    '''
    def __init__(self, num_in_ch, num_out_ch, upscale, num_block, num_feat, num_residual_feat):
        super().__init__()
        assert (num_feat > num_residual_feat >= num_in_ch) and (num_out_ch == num_in_ch)

        num_feat = num_feat - num_residual_feat

        self.conv_first = nn.Conv2d(num_in_ch, num_feat, kernel_size=3, padding=1)
        self.conv_residual_first = nn.Conv2d(num_in_ch, num_residual_feat, kernel_size=3, padding=1)

        backbone_convs = []
        residual_convs = []
        add_residual_convs = []
        for _ in range(num_block):
            backbone_convs.append(nn.Conv2d(num_feat, num_feat, 3, padding=1))
            residual_convs.append(nn.Conv2d(num_residual_feat, num_residual_feat, kernel_size=3, padding=1))
            add_residual_convs.append(nn.Conv2d(num_residual_feat, num_feat, kernel_size=3, padding=1))
        self.backbone_convs = nn.ModuleList(backbone_convs)
        self.residual_convs = nn.ModuleList(residual_convs)
        self.add_residual_convs = nn.ModuleList(add_residual_convs)

        self.conv_last = nn.Conv2d(num_feat, num_out_ch * (upscale**2), 3, padding=1)
        self.conv_residual_last = nn.Conv2d(num_residual_feat, num_out_ch * (upscale**2), kernel_size=3, padding=1)

        self.upsampler = nn.PixelShuffle(upscale)

        self.init_weights(num_in_ch, upscale, num_residual_feat)

    def init_weights(self, colors, upscale, num_residual_feat):
        ''' init weights (K_r --> I, K_{r2b} --> O, K_{b2r} --> O, and repeat --> repeat(I,n)) '''
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
        self.conv_residual_last.weight.data.fill_(0)
        for i in range(colors):
            for j in range(upscale**2):
                self.conv_residual_last.weight.data[i * (upscale**2) + j, i, 1, 1] = 1
        self.conv_residual_last.bias.data.fill_(0)

    def forward(self, input):
        ''' forward '''
        x = F.relu(self.conv_first(input))
        r = F.relu(self.conv_residual_first(input))

        for backbone_conv, residual_conv, add_residual_conv in zip(self.backbone_convs, self.residual_convs, self.add_residual_convs):
            x, r = F.relu(backbone_conv(x) + add_residual_conv(r)), F.relu(residual_conv(r))

        x = self.upsampler(self.conv_last(x))
        r = self.upsampler(self.conv_residual_last(r))
        return x + r, r


class ETDSForInference(nn.Module):
    ''' ETDS for inference arch '''
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
            # Reason for +1: extra channels are added to ensure that the number of channels is a multiple of 4.
            self.conv_last = nn.Conv2d(num_feat, num_out_ch * (upscale**2) + 1, kernel_size=3, padding=1)
            self.conv_clip = nn.Conv2d(num_out_ch * (upscale**2) + 1, num_out_ch * (upscale**2) + 1, 1)
        else:
            self.conv_last = nn.Conv2d(num_feat, num_out_ch * (upscale**2), kernel_size=3, padding=1)
            self.conv_clip = nn.Conv2d(num_out_ch * (upscale**2), num_out_ch * (upscale**2), 1)

        self.upsampler = nn.PixelShuffle(upscale)

    def forward(self, input):
        ''' forward '''
        x = F.relu(self.conv_first(input))

        for backbone_conv in self.backbone_convs:
            x = F.relu(backbone_conv(x))

        x = F.relu(self.conv_last(x))
        if self.upscale == 3:
            # Reason: extra channels are added to ensure that the number of channels is a multiple of 4.
            x = F.relu(self.conv_clip(x))[:, :-1, :, :]
        else:
            x = F.relu(self.conv_clip(x))

        x = self.upsampler(x)
        return x


class ETDSOnlyET(nn.Module):
    ''' ETDSOnlyET (Ablation) '''
    def __init__(self, num_in_ch, num_out_ch, upscale, num_block, num_feat, num_residual_feat):
        super().__init__()
        assert (num_feat > num_residual_feat >= num_in_ch) and (num_out_ch == num_in_ch)

        self.upscale = upscale
        num_feat = num_feat - num_residual_feat

        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, padding=1)

        backbone_convs = []
        for _ in range(num_block):
            backbone_convs.append(nn.Conv2d(num_feat, num_feat, 3, padding=1))
        self.backbone_convs = nn.ModuleList(backbone_convs)

        self.conv_last = nn.Conv2d(num_feat, num_out_ch * (upscale**2), 3, padding=1)

        self.upsampler = nn.PixelShuffle(upscale)

    def forward(self, input):
        ''' forward '''
        shortcut = torch.repeat_interleave(input, self.upscale * self.upscale, dim=1)
        x = F.relu(self.conv_first(input))
        for backbone_conv in self.backbone_convs:
            x = F.relu(backbone_conv(x))

        x = self.upsampler(self.conv_last(x) + shortcut)
        return x


class ETDSAblationResidual(nn.Module):
    ''' ETDSAblationResidual (Ablation) '''
    def __init__(self, num_in_ch, num_out_ch, upscale, num_block, num_feat, num_residual_feat):
        super().__init__()
        assert (num_feat > num_residual_feat >= num_in_ch) and (num_out_ch == num_in_ch)

        num_feat = num_feat - num_residual_feat

        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, padding=1)
        self.conv_residual_first = nn.Conv2d(num_in_ch, num_residual_feat, kernel_size=3, padding=1)

        backbone_convs = []
        residual_convs = []
        for _ in range(num_block):
            backbone_convs.append(nn.Conv2d(num_feat, num_feat, 3, padding=1))
            residual_convs.append(nn.Conv2d(num_residual_feat, num_residual_feat, kernel_size=3, padding=1))
        self.backbone_convs = nn.ModuleList(backbone_convs)
        self.residual_convs = nn.ModuleList(residual_convs)

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
        self.conv_residual_last.weight.data.fill_(0)
        for i in range(colors):
            for j in range(upscale**2):
                self.conv_residual_last.weight.data[i * (upscale**2) + j, i, 1, 1] = 1
        self.conv_residual_last.bias.data.fill_(0)

    def forward(self, input):
        ''' forward '''
        x = F.relu(self.conv_first(input))
        r = F.relu(self.conv_residual_first(input))

        for backbone_conv, residual_conv in zip(self.backbone_convs, self.residual_convs):
            x, r = F.relu(backbone_conv(x)), F.relu(residual_conv(r))

        x = self.upsampler(self.conv_last(x))
        r = self.upsampler(self.conv_residual_last(r))
        return x + r, r


class ETDSAblationAdvancedResidual(nn.Module):
    ''' ETDSAblationAdvancedResidual (Ablation) '''
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

    def forward(self, input):
        ''' forward '''
        x = F.relu(self.conv_first(input))
        r = F.relu(self.conv_residual_first(input))

        for backbone_conv, residual_conv, add_residual_conv, advanced_residual_conv in zip(self.backbone_convs, self.residual_convs, self.add_residual_convs, self.advanced_residual_convs):
            x, r = F.relu(backbone_conv(x) + add_residual_conv(r)), F.relu(residual_conv(r) + advanced_residual_conv(x))

        x = self.upsampler(self.conv_last(x))
        r = self.upsampler(self.conv_residual_last(r))
        return x + r, r
