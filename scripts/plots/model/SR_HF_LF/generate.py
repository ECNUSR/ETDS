''' separate high-frequency and low-frequency components in the super-resolution process. '''
import cv2
import torch
from torch import nn
from torch.nn import functional as F


class ETDS(nn.Module):
    ''' ETDS '''
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

    def forward(self, input):
        ''' forward '''
        x = F.relu(self.conv_first(input))
        r = F.relu(self.conv_residual_first(input))

        for backbone_conv, residual_conv, add_residual_conv in zip(self.backbone_convs, self.residual_convs, self.add_residual_convs):
            x, r = F.relu(backbone_conv(x) + add_residual_conv(r)), F.relu(residual_conv(r))

        x = self.upsampler(self.conv_last(x))
        r = self.upsampler(self.conv_residual_last(r))
        return x, r


def main():
    ''' main '''
    model = ETDS(3, 3, 3, 7, 48, 3)
    model.load_state_dict(torch.load('scripts/plots/model/SR_HF_LF/ETDS_M7C48_x3_no_convert.pth'))

    input = torch.from_numpy(cv2.cvtColor(cv2.imread('scripts/plots/model/SR_HF_LF/doggy_x3.png'), cv2.COLOR_BGR2RGB) / 255.0).float().permute(2, 0, 1).contiguous().unsqueeze(0)

    with torch.no_grad():
        HF, LF = model(input)
        SR = HF + LF
    HF = cv2.cvtColor(HF.squeeze().permute(1, 2, 0).numpy() * 255, cv2.COLOR_RGB2BGR) * 28 # 乘以2是为了增强视觉效果
    LF = cv2.cvtColor(LF.squeeze().permute(1, 2, 0).numpy() * 255, cv2.COLOR_RGB2BGR)
    SR = cv2.cvtColor(SR.squeeze().permute(1, 2, 0).numpy() * 255, cv2.COLOR_RGB2BGR)

    cv2.imwrite('scripts/plots/model/SR_HF_LF/LF.png', LF)
    cv2.imwrite('scripts/plots/model/SR_HF_LF/HF.png', HF)
    cv2.imwrite('scripts/plots/model/SR_HF_LF/SR.png', SR)


if __name__ == '__main__':
    main()
