''' ESPCN '''
from torch import nn
from torch.nn import functional as F


class ESPCN(nn.Module):
    ''' ESPCN '''
    def __init__(self, colors, upscale):
        super().__init__()

        self.conv1 = nn.Conv2d(colors, 64, 5, padding=2)
        self.conv2 = nn.Conv2d(64, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, colors * (upscale ** 2), 3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(upscale)

    def forward(self, x):
        ''' forward '''
        x = F.tanh(self.conv1(x))
        x = F.tanh(self.conv2(x))
        x = F.sigmoid(self.pixel_shuffle(self.conv3(x)))
        return x
