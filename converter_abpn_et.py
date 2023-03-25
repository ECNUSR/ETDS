''' converter '''
import math
import argparse
import os
from typing import OrderedDict
import torch
from core.archs.ir import ABPN, ABPN_ET


def get_info(state: OrderedDict) -> dict:
    ''' Obtain information such as the number of layers and channels of the model according to the checkpoint file. '''
    # num_feat, num_block, scale
    num_feat = state['backbone.0.weight'].shape[0]
    num_block = len([name for name in state if 'weight' in name]) - 3
    scale = round(math.sqrt(state[[name for name in state if 'weight' in name][-1]].shape[0] // 3))
    return {
        'num_feat': num_feat,
        'num_block': num_block,
        'scale': scale,
    }


def model_allclose(model1, model2):
    ''' Determine whether the output of the model is similar '''
    # Randomly generate input
    input = torch.rand(4, 3, 64, 64)

    out1 = torch.clamp(model1(input), min=0, max=1)
    out2 = model2(input)    # No clip operation added for model2 because the logic is already embedded.

    return torch.max((out1 - out2).abs()).item() * 255 < 0.01


def convert(state: OrderedDict) -> OrderedDict:
    ''' Convert ETDS to plain network '''
    # Get the hyperparameters of the model according to the checkpoint
    info = get_info(state)
    model1 = ABPN(**info)
    model1.load_state_dict(state)
    model2 = ABPN_ET(**info)

    # for conv_first (Eq. 9)
    model2.backbone[0].weight.data[:-4, :, :, :] = model1.backbone[0].weight.data
    model2.backbone[0].bias.data[:-4] = model1.backbone[0].bias.data
    model2.backbone[0].weight.data[-4:, :, :, :] = 0
    model2.backbone[0].bias.data[-4:] = 0
    for i in range(3):
        model2.backbone[0].weight.data[i - 3, i - 3, 1, 1] = 1

    # for backbones (Eq. 8)
    for i in range(info['num_block']):
        model2.backbone[2 + i][0].weight.data[:, :, :, :] = 0
        model2.backbone[2 + i][0].bias.data[:] = 0
        model2.backbone[2 + i][0].weight.data[:-4, :-4, :, :] = model1.backbone[2 + i][0].weight.data
        model2.backbone[2 + i][0].bias.data[:-4] = model1.backbone[2 + i][0].bias.data
        for j in range(3):
            model2.backbone[2 + i][0].weight.data[j - 3, j - 3, 1, 1] = 1

    # for conv_last (Eq. 5, Eq. 6, Eq. 7, Eq. 12 and reparameterization)
    weight = model1.backbone[2 + info['num_block']].weight.data
    model2.backbone[2 + info['num_block']].weight.data[:, :, :, :] = 0
    model2.backbone[2 + info['num_block']].bias.data[:] = 0
    model2.backbone[2 + info['num_block']].weight.data[:weight.shape[0], :weight.shape[1], :, :] = model1.backbone[2 + info['num_block']].weight.data
    model2.backbone[2 + info['num_block']].bias.data[:weight.shape[0]] = model1.backbone[2 + info['num_block']].bias.data
    for i in range(3):
        model2.backbone[2 + info['num_block']].weight.data[i - 3, i - 3, 1, 1] = 1
    weight = model1.backbone[4 + info['num_block']].weight.data
    model2.backbone[4 + info['num_block']].weight.data[:, :, :, :] = 0
    model2.backbone[4 + info['num_block']].bias.data[:] = 1
    model2.backbone[4 + info['num_block']].weight.data[:weight.shape[0], :weight.shape[1], :, :] = -model1.backbone[4 + info['num_block']].weight.data
    model2.backbone[4 + info['num_block']].bias.data[:weight.shape[0]] = 1 - model1.backbone[4 + info['num_block']].bias.data
    for i in range(3):
        for j in range(info['scale'] ** 2):
            model2.backbone[4 + info['num_block']].weight.data[i * (info['scale'] ** 2) + j, i - 3, 1, 1] = -1

    # for conv_clip (Eq. 12)
    model2.backbone[6 + info['num_block']].weight.data[:, :, :, :] = 0
    model2.backbone[6 + info['num_block']].bias.data[:] = 1
    for i in range(model2.backbone[6 + info['num_block']].weight.shape[0]):
        model2.backbone[6 + info['num_block']].weight.data[i, i , 0, 0] = -1

    # Verify that the network output before and after the transformation is the same (due to floating point numbers, etc., it cannot be exactly the same).
    assert model_allclose(model1, model2)

    return model2.state_dict()


def main():
    ''' main '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='ABPN model state path.')
    parser.add_argument('--output', type=str, required=True, help='ABPN for inference output path.')
    args = parser.parse_args()

    if os.path.split(args.output)[0] != '':
        os.makedirs(os.path.split(args.output)[0], exist_ok=True)

    state = torch.load(args.input)
    converted_state = convert(state)
    torch.save(converted_state, args.output)


if __name__ == '__main__':
    main()
