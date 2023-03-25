''' FSRCNN '''
import math
from torch import nn


class FSRCNN(nn.Module):
    ''' FSRCNN '''
    def __init__(self, upscale):
        super().__init__()
        self.first_part = nn.Sequential(
            nn.Conv2d(3, 56, kernel_size=5, padding=5//2),
            nn.PReLU(56)
        )
        self.mid_part = [nn.Conv2d(56, 12, kernel_size=1), nn.PReLU(12)]
        for _ in range(4):
            self.mid_part.extend([nn.Conv2d(12, 12, kernel_size=3, padding=3//2), nn.PReLU(12)])
        self.mid_part.extend([nn.Conv2d(12, 56, kernel_size=1), nn.PReLU(56)])
        self.mid_part = nn.Sequential(*self.mid_part)
        self.last_part = nn.ConvTranspose2d(56, 3, kernel_size=9, stride=upscale, padding=9//2, output_padding=upscale-1)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.first_part:
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, mean=0.0, std=math.sqrt(2/(m.out_channels*m.weight.data[0][0].numel())))
                nn.init.zeros_(m.bias.data)
        for m in self.mid_part:
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, mean=0.0, std=math.sqrt(2/(m.out_channels*m.weight.data[0][0].numel())))
                nn.init.zeros_(m.bias.data)
        nn.init.normal_(self.last_part.weight.data, mean=0.0, std=0.001)
        nn.init.zeros_(self.last_part.bias.data)

    def forward(self, x):
        ''' forward '''
        x = self.first_part(x)
        x = self.mid_part(x)
        x = self.last_part(x)
        return x
