''' echo params '''
import argparse
import torch


def main():
    ''' main '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, type=str, help='model')
    args = parser.parse_args()

    print(sum(p.numel() for _, p in torch.load(args.model).items()) / 1000)

if __name__ == '__main__':
    main()
