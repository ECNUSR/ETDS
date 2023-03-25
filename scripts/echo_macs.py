''' calc macs '''
import sys
import argparse
import torch
import yaml
from thop import profile
try:
    from core.archs import build_network
    from core.utils.options import ordered_yaml
except Exception:
    sys.path.append(sys.path[0].replace('scripts', ''))
    from core.archs import build_network
    from core.utils.options import ordered_yaml


def main():
    ''' main '''
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', required=True, type=str, help='config path')
    parser.add_argument('--color', type=int, default=3, help='color')
    parser.add_argument('--height', type=int, default=360, help='height')
    parser.add_argument('--width', type=int, default=640, help='width')
    args = parser.parse_args()

    with open(args.opt, mode='r') as f:
        opt = yaml.load(f, Loader=ordered_yaml())

    input = torch.rand(1, args.color, args.height, args.width)

    model = build_network(opt['network_g'])

    macs, params = profile(model, inputs=(input, ))

    print(f"macs: {macs / 1e9:.4f} G, params: {params / 1e3:.4f} K")


if __name__ == '__main__':
    main()
