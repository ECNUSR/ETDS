''' converter '''
import math
import argparse
import os
from typing import OrderedDict
import torch
from core.archs.ir import ECBSRT, ECBSRET


def get_info(state: OrderedDict) -> dict:
    ''' Obtain information such as the number of layers and channels of the model according to the checkpoint file. '''
    # num_in_ch, num_out_ch, upscale, num_block, num_feat,
    num_in_ch = num_out_ch = state['backbone.0.0.weight'].shape[1]
    num_feat = state['backbone.0.0.weight'].shape[0]
    num_block = (len([name for name in state if 'backbone.' in name]) + 1) // 3 - 2
    upscale = round(math.sqrt(state[[name for name in state if '.weight' in name][-1]].shape[0] // num_out_ch))
    return {
        'num_in_ch': num_in_ch,
        'num_out_ch': num_out_ch,
        'num_feat': num_feat,
        'num_block': num_block,
        'upscale': upscale,
        'act_type': 'prelu',        # assert act_type = prelu
    }


def model_allclose(model1, model2, num_in_ch):
    ''' Determine whether the output of the model is similar '''
    # Randomly generate input
    input = torch.rand(4, num_in_ch, 64, 64)

    out1 = torch.clamp(model1(input), min=0, max=1)
    out2 = model2(input)    # No clip operation added for model2 because the logic is already embedded.

    return torch.max((out1 - out2).abs()).item() * 255 < 0.01


def convert(state: OrderedDict) -> OrderedDict:
    ''' Convert ECBSR to plain network '''
    # Get the hyperparameters of the model according to the checkpoint
    info = get_info(state)
    # Create model objects based on hyperparameters and load checkpoints.
    model1 = ECBSRT(**info)
    model1.load_state_dict(state)
    model2 = ECBSRET(**info)

    # for conv_first (Eq. 9)
    model2.backbone[0][0].weight.data[:, :, :, :] = 0
    model2.backbone[0][0].bias.data[:] = 0
    model2.backbone[0][1].weight.data[:] = 0
    model2.backbone[0][0].weight.data[:-4, :, :, :] = model1.backbone[0][0].weight.data
    model2.backbone[0][0].bias.data[:-4] = model1.backbone[0][0].bias.data
    model2.backbone[0][1].weight.data[:-4] = model1.backbone[0][1].weight.data  # for prelu
    for i in range(info['num_in_ch']):
        model2.backbone[0][0].weight.data[i - 3, i - 3, 1, 1] = 1

    # for backbones (Eq. 8)
    for i in range(info['num_block']):
        model2.backbone[1 + i][0].weight.data[:, :, :, :] = 0
        model2.backbone[1 + i][0].bias.data[:] = 0
        model2.backbone[1 + i][1].weight.data[:] = 0
        model2.backbone[1 + i][0].weight.data[:-4, :-4, :, :] = model1.backbone[1 + i][0].weight.data
        model2.backbone[1 + i][0].bias.data[:-4] = model1.backbone[1 + i][0].bias.data
        model2.backbone[1 + i][1].weight.data[:-4] = model1.backbone[1 + i][1].weight.data  # for prelu
        for j in range(info['num_in_ch']):
            model2.backbone[1 + i][0].weight.data[j - 3, j - 3, 1, 1] = 1

    # for conv_last (Eq. 5, Eq. 6, Eq. 7, Eq. 12 and reparameterization)
    weight = model1.backbone[1 + info['num_block']][0].weight.data
    model2.backbone[1 + info['num_block']][0].weight.data[:, :, :, :] = 0
    model2.backbone[1 + info['num_block']][0].bias.data[:] = 1
    model2.backbone[1 + info['num_block']][0].weight.data[:weight.shape[0], :weight.shape[1], :, :] = -model1.backbone[1 + info['num_block']][0].weight.data
    model2.backbone[1 + info['num_block']][0].bias.data[:weight.shape[0]] = 1 - model1.backbone[1 + info['num_block']][0].bias.data
    for i in range(info['num_in_ch']):
        for j in range(info['upscale'] ** 2):
            model2.backbone[1 + info['num_block']][0].weight.data[i * (info['upscale'] ** 2) + j, i - 3, 1, 1] = -1

    # for conv_clip (Eq. 12)
    model2.backbone[2 + info['num_block']][0].weight.data[:, :, :, :] = 0
    model2.backbone[2 + info['num_block']][0].bias.data[:] = 1
    for i in range(model2.backbone[2 + info['num_block']][0].weight.shape[0]):
        model2.backbone[2 + info['num_block']][0].weight.data[i, i , 0, 0] = -1

    # Verify that the network output before and after the transformation is the same (due to floating point numbers, etc., it cannot be exactly the same).
    assert model_allclose(model1, model2, info['num_in_ch'])

    return model2.state_dict()


def main():
    ''' main '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='ECBSR model state path.')
    parser.add_argument('--output', type=str, required=True, help='ECBSR for inference output path.')
    args = parser.parse_args()

    if os.path.split(args.output)[0] != '':
        os.makedirs(os.path.split(args.output)[0], exist_ok=True)

    state = torch.load(args.input)
    converted_state = convert(state)
    torch.save(converted_state, args.output)


if __name__ == '__main__':
    main()
