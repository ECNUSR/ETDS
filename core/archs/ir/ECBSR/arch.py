''' ECBSR arch '''
import torch
from torch import nn
from torch.nn import functional as F


class SeqConv3x3(nn.Module):
    ''' SeqConv3x3 '''
    def __init__(self, seq_type, inp_planes, out_planes, depth_multiplier):
        super().__init__()

        self.type = seq_type
        self.out_planes = out_planes

        if self.type == 'conv1x1-conv3x3':
            self.mid_planes = int(out_planes * depth_multiplier)
            conv0 = torch.nn.Conv2d(inp_planes,
                                    self.mid_planes,
                                    kernel_size=1,
                                    padding=0)
            self.k0 = conv0.weight
            self.b0 = conv0.bias
            conv1 = torch.nn.Conv2d(self.mid_planes,
                                    self.out_planes,
                                    kernel_size=3)
            self.k1 = conv1.weight
            self.b1 = conv1.bias
        elif self.type == 'conv1x1-sobelx':
            conv0 = torch.nn.Conv2d(inp_planes,
                                    self.out_planes,
                                    kernel_size=1,
                                    padding=0)
            self.k0 = conv0.weight
            self.b0 = conv0.bias
            scale = torch.randn(size=(self.out_planes, 1, 1, 1)) * 1e-3
            self.scale = nn.Parameter(scale)
            bias = torch.randn(self.out_planes) * 1e-3
            bias = torch.reshape(bias, (self.out_planes, ))
            self.bias = nn.Parameter(bias)
            self.mask = torch.zeros((self.out_planes, 1, 3, 3),
                                    dtype=torch.float32)
            for i in range(self.out_planes):
                self.mask[i, 0, 0, 0] = 1.0
                self.mask[i, 0, 1, 0] = 2.0
                self.mask[i, 0, 2, 0] = 1.0
                self.mask[i, 0, 0, 2] = -1.0
                self.mask[i, 0, 1, 2] = -2.0
                self.mask[i, 0, 2, 2] = -1.0
            self.mask = nn.Parameter(data=self.mask, requires_grad=False)
        elif self.type == 'conv1x1-sobely':
            conv0 = torch.nn.Conv2d(inp_planes,
                                    self.out_planes,
                                    kernel_size=1,
                                    padding=0)
            self.k0 = conv0.weight
            self.b0 = conv0.bias
            scale = torch.randn(size=(self.out_planes, 1, 1, 1)) * 1e-3
            self.scale = nn.Parameter(torch.FloatTensor(scale))
            bias = torch.randn(self.out_planes) * 1e-3
            bias = torch.reshape(bias, (self.out_planes, ))
            self.bias = nn.Parameter(torch.FloatTensor(bias))
            self.mask = torch.zeros((self.out_planes, 1, 3, 3),
                                    dtype=torch.float32)
            for i in range(self.out_planes):
                self.mask[i, 0, 0, 0] = 1.0
                self.mask[i, 0, 0, 1] = 2.0
                self.mask[i, 0, 0, 2] = 1.0
                self.mask[i, 0, 2, 0] = -1.0
                self.mask[i, 0, 2, 1] = -2.0
                self.mask[i, 0, 2, 2] = -1.0
            self.mask = nn.Parameter(data=self.mask, requires_grad=False)
        elif self.type == 'conv1x1-laplacian':
            conv0 = torch.nn.Conv2d(inp_planes,
                                    self.out_planes,
                                    kernel_size=1,
                                    padding=0)
            self.k0 = conv0.weight
            self.b0 = conv0.bias
            scale = torch.randn(size=(self.out_planes, 1, 1, 1)) * 1e-3
            self.scale = nn.Parameter(torch.FloatTensor(scale))
            bias = torch.randn(self.out_planes) * 1e-3
            bias = torch.reshape(bias, (self.out_planes, ))
            self.bias = nn.Parameter(torch.FloatTensor(bias))
            self.mask = torch.zeros((self.out_planes, 1, 3, 3),
                                    dtype=torch.float32)
            for i in range(self.out_planes):
                self.mask[i, 0, 0, 1] = 1.0
                self.mask[i, 0, 1, 0] = 1.0
                self.mask[i, 0, 1, 2] = 1.0
                self.mask[i, 0, 2, 1] = 1.0
                self.mask[i, 0, 1, 1] = -4.0
            self.mask = nn.Parameter(data=self.mask, requires_grad=False)
        else:
            raise ValueError('the type of seqconv is not supported!')

    def forward(self, x):
        ''' forward '''
        if self.type == 'conv1x1-conv3x3':
            y0 = F.conv2d(input=x, weight=self.k0, bias=self.b0, stride=1)
            y0 = F.pad(y0, (1, 1, 1, 1), 'constant', 0)
            b0_pad = self.b0.view(1, -1, 1, 1)
            y0[:, :, 0:1, :] = b0_pad
            y0[:, :, -1:, :] = b0_pad
            y0[:, :, :, 0:1] = b0_pad
            y0[:, :, :, -1:] = b0_pad
            y1 = F.conv2d(input=y0, weight=self.k1, bias=self.b1, stride=1)
        else:
            y0 = F.conv2d(input=x, weight=self.k0, bias=self.b0, stride=1)
            y0 = F.pad(y0, (1, 1, 1, 1), 'constant', 0)
            b0_pad = self.b0.view(1, -1, 1, 1)
            y0[:, :, 0:1, :] = b0_pad
            y0[:, :, -1:, :] = b0_pad
            y0[:, :, :, 0:1] = b0_pad
            y0[:, :, :, -1:] = b0_pad
            y1 = F.conv2d(input=y0,
                          weight=self.scale * self.mask,
                          bias=self.bias,
                          stride=1,
                          groups=self.out_planes)
        return y1

    def rep_params(self):
        ''' rep_params '''
        device = self.k0.get_device()
        if device < 0:
            device = None
        if self.type == 'conv1x1-conv3x3':
            RK = F.conv2d(input=self.k1, weight=self.k0.permute(1, 0, 2, 3))
            RB = torch.ones(1, self.mid_planes, 3, 3,
                            device=device) * self.b0.view(1, -1, 1, 1)
            RB = F.conv2d(input=RB, weight=self.k1).view(-1, ) + self.b1
        else:
            tmp = self.scale * self.mask
            k1 = torch.zeros((self.out_planes, self.out_planes, 3, 3),
                             device=device)
            for i in range(self.out_planes):
                k1[i, i, :, :] = tmp[i, 0, :, :]
            b1 = self.bias
            RK = F.conv2d(input=k1, weight=self.k0.permute(1, 0, 2, 3))
            RB = torch.ones(1, self.out_planes, 3, 3,
                            device=device) * self.b0.view(1, -1, 1, 1)
            RB = F.conv2d(input=RB, weight=k1).view(-1, ) + b1
        return RK, RB


class ECB(nn.Module):
    ''' ECB block '''
    def __init__(self,
                 inp_planes,
                 out_planes,
                 depth_multiplier,
                 act_type='prelu'):
        super().__init__()

        self.conv3x3 = torch.nn.Conv2d(inp_planes,
                                       out_planes,
                                       kernel_size=3,
                                       padding=1)
        self.conv1x1_3x3 = SeqConv3x3('conv1x1-conv3x3', inp_planes,
                                      out_planes, depth_multiplier)
        self.conv1x1_sbx = SeqConv3x3('conv1x1-sobelx', inp_planes, out_planes,
                                      -1)
        self.conv1x1_sby = SeqConv3x3('conv1x1-sobely', inp_planes, out_planes,
                                      -1)
        self.conv1x1_lpl = SeqConv3x3('conv1x1-laplacian', inp_planes,
                                      out_planes, -1)

        self.act = self.get_activation(act_type, out_planes)

    @staticmethod
    def get_activation(act_type, channels):
        ''' get activation '''
        if act_type == 'prelu':
            return nn.PReLU(num_parameters=channels)
        if act_type == 'relu':
            return nn.ReLU(inplace=True)
        if act_type == 'rrelu':
            return nn.RReLU(lower=-0.05, upper=0.05)
        if act_type == 'softplus':
            return nn.Softplus()
        if act_type == 'linear':
            return nn.Identity()
        raise ValueError('The type of activation if not support!')

    def forward(self, x):
        ''' forward '''
        if self.training:
            y = self.conv3x3(x) + self.conv1x1_3x3(x) + self.conv1x1_sbx(
                x) + self.conv1x1_sby(x) + self.conv1x1_lpl(x)
        else:
            RK, RB = self.rep_params()
            y = F.conv2d(input=x, weight=RK, bias=RB, stride=1, padding=1)
        y = self.act(y)
        return y

    def rep_params(self):
        ''' rep params '''
        K0, B0 = self.conv3x3.weight, self.conv3x3.bias
        K1, B1 = self.conv1x1_3x3.rep_params()
        K2, B2 = self.conv1x1_sbx.rep_params()
        K3, B3 = self.conv1x1_sby.rep_params()
        K4, B4 = self.conv1x1_lpl.rep_params()
        RK, RB = (K0 + K1 + K2 + K3 + K4), (B0 + B1 + B2 + B3 + B4)
        return RK, RB


class ECBSR(nn.Module):
    ''' ECBSR '''
    def __init__(self, num_in_ch, num_out_ch, upscale, num_block, num_feat,
                 act_type):
        super().__init__()
        self.num_in_ch = num_in_ch
        self.upscale = upscale

        backbone = []
        backbone += [
            ECB(num_in_ch, num_feat, depth_multiplier=2.0, act_type=act_type)
        ]
        for _ in range(num_block):
            backbone += [
                ECB(num_feat,
                    num_feat,
                    depth_multiplier=2.0,
                    act_type=act_type)
            ]
        backbone += [
            ECB(num_feat,
                num_out_ch * (upscale**2),
                depth_multiplier=2.0,
                act_type='linear')
        ]

        self.backbone = nn.Sequential(*backbone)
        self.upsampler = nn.PixelShuffle(upscale)

    def forward(self, x):
        ''' forward '''
        if self.num_in_ch > 1:
            shortcut = torch.repeat_interleave(x, self.upscale * self.upscale, dim=1)
        else:
            shortcut = x
        y = self.backbone(x) + shortcut
        y = self.upsampler(y)
        return y


class ECBSRT(nn.Module):
    ''' ECBSR Test '''
    def __init__(self, num_in_ch, num_out_ch, upscale, num_block, num_feat,
                 act_type):
        super().__init__()
        self.num_in_ch = num_in_ch
        self.upscale = upscale

        backbone = [
            nn.Sequential(
                nn.Conv2d(num_in_ch, num_feat, 3, padding=1),
                ECB.get_activation(act_type, num_feat),
            )
        ]
        for _ in range(num_block):
            backbone.append(
                nn.Sequential(nn.Conv2d(num_feat, num_feat, 3, padding=1),
                              ECB.get_activation(act_type, num_feat)))
        backbone.append(
            nn.Sequential(
                nn.Conv2d(num_feat, num_out_ch * (upscale**2), 3, padding=1)))

        self.backbone = nn.Sequential(*backbone)
        self.upsampler = nn.PixelShuffle(upscale)

    def forward(self, x):
        ''' forward '''
        if self.num_in_ch > 1:
            shortcut = torch.repeat_interleave(x, self.upscale * self.upscale, dim=1)
        else:
            shortcut = x
        y = self.backbone(x) + shortcut
        y = self.upsampler(y)
        return y


class ECBSRET(nn.Module):
    ''' ECBSR for ET '''
    def __init__(self, num_in_ch, num_out_ch, upscale, num_block, num_feat, act_type):
        super().__init__()
        # Reason for +4 instead of +3: extra channels are added to ensure that the number of channels is a multiple of 4. !!!!!
        self.upscale = upscale
        backbone = [
            nn.Sequential(
                nn.Conv2d(num_in_ch, num_feat + 4, 3, padding=1),   # +4 for speed, but ugly
                ECB.get_activation(act_type, num_feat + 4),
            )
        ]
        for _ in range(num_block):
            backbone.append(
                nn.Sequential(
                    nn.Conv2d(num_feat + 4, num_feat + 4, 3, padding=1),
                    ECB.get_activation(act_type, num_feat + 4)
                )
            )
        if upscale == 3:
            backbone.append(
                nn.Sequential(
                    nn.Conv2d(num_feat + 4, num_out_ch * (upscale**2) + 1, 3, padding=1),
                    nn.ReLU(),
                )
            )
            backbone.append(
                nn.Sequential(
                    nn.Conv2d(num_out_ch * (upscale**2) + 1, num_out_ch * (upscale**2) + 1, 1),
                    nn.ReLU(),
                )
            )
        else:
            backbone.append(
                nn.Sequential(
                    nn.Conv2d(num_feat + 4, num_out_ch * (upscale**2), 3, padding=1),
                    nn.ReLU(),
                )
            )
            backbone.append(
                nn.Sequential(
                    nn.Conv2d(num_out_ch * (upscale**2), num_out_ch * (upscale**2), 1),
                    nn.ReLU(),
                )
            )

        self.backbone = nn.Sequential(*backbone)
        self.upsampler = nn.PixelShuffle(upscale)

    def forward(self, x):
        ''' forward '''
        y = self.backbone(x)
        if self.upscale == 3:
            y = self.upsampler(y[:, :-1, :, :])
        else:
            y = self.upsampler(y)
        return y
