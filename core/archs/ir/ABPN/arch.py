''' ABPN '''
import torch
from torch import nn


class ABPN(nn.Module):
    ''' ABPN '''
    def __init__(self, num_feat, num_block, scale) -> None:
        super().__init__()
        self.scale = scale
        self.backbone = nn.Sequential(
            nn.Conv2d(3, num_feat, 3, padding=1),
            nn.ReLU(),
            *[nn.Sequential(nn.Conv2d(num_feat, num_feat, 3, padding=1), nn.ReLU()) for _ in range(num_block)],
            nn.Conv2d(num_feat, (scale**2)*3, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d((scale**2)*3, (scale**2)*3, 3, padding=1),
        )

        self.upsampler = nn.PixelShuffle(scale)

    def forward(self, input):
        ''' forward '''
        shortcut = torch.repeat_interleave(input, self.scale * self.scale, dim=1)
        y = self.backbone(input) + shortcut
        y = self.upsampler(y)
        return y


class ABPN_ET(nn.Module):
    ''' ABPN_ET '''
    def __init__(self, num_feat, num_block, scale) -> None:
        super().__init__()
        # Reason for +4 instead of +3: extra channels are added to ensure that the number of channels is a multiple of 4. !!!!
        self.scale = scale

        if scale == 3:
            tail_blocks = [
                nn.Conv2d(num_feat + 4, (scale**2)*3 + 3, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d((scale**2)*3 + 3, (scale**2)*3 + 1, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d((scale**2)*3 + 1, (scale**2)*3 + 1, 1),
                nn.ReLU()
            ]
        else:
            tail_blocks = [
                nn.Conv2d(num_feat + 4, (scale**2)*3 + 4, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d((scale**2)*3 + 4, (scale**2)*3, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d((scale**2)*3, (scale**2)*3, 1),
                nn.ReLU()
            ]

        self.backbone = nn.Sequential(
            nn.Conv2d(3, num_feat + 4, 3, padding=1),   # +4 for speed, but ugly
            nn.ReLU(),
            *[nn.Sequential(nn.Conv2d(num_feat + 4, num_feat + 4, 3, padding=1), nn.ReLU()) for _ in range(num_block)],
            *tail_blocks
        )

        self.upsampler = nn.PixelShuffle(scale)

    def forward(self, input):
        ''' forward '''
        y = self.backbone(input)
        if self.scale == 3:
            y = self.upsampler(y[:, :-1, :, :])
        else:
            y = self.upsampler(y)
        return y
