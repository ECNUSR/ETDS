''' Plain arch '''
from torch import nn
from torch.nn import functional as F


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

    def forward(self, input):
        ''' forward '''
        x = F.relu(self.conv_first(input))

        for backbone_conv in self.backbone_convs:
            x = F.relu(backbone_conv(x))

        x = self.upsampler(self.conv_last(x))
        return x
