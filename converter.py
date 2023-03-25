''' converter '''
import math
import argparse
import os
from typing import OrderedDict
import torch
from core.archs.ir import ETDS, ETDSForInference


def get_info(state: OrderedDict) -> dict:
    ''' Obtain information such as the number of layers and channels of the model according to the checkpoint file. '''
    num_in_ch = num_out_ch = state['conv_first.weight'].shape[1]
    num_feat = state['conv_first.weight'].shape[0] + num_in_ch
    num_block = len([name for name in state if 'backbone_convs.' in name]) // 2
    num_residual_feat = state['conv_residual_first.weight'].shape[0]
    upscale = round(math.sqrt(state['conv_last.weight'].shape[0] // num_out_ch))
    return {
        'num_in_ch': num_in_ch,
        'num_out_ch': num_out_ch,
        'num_feat': num_feat,
        'num_block': num_block,
        'num_residual_feat': num_residual_feat,
        'upscale': upscale,
    }


def model_allclose(model1: ETDS, model2: ETDSForInference, num_in_ch: int) -> bool:
    ''' Determine whether the output of the model is similar '''
    # Randomly generate input
    input = torch.rand(4, num_in_ch, 64, 64)

    out1 = torch.clamp(model1(input)[0], min=0, max=1)
    out2 = model2(input)    # No clip operation added for model2 because the logic is already embedded.

    return torch.max((out1 - out2).abs()).item() * 255 < 0.01


def convert(state: OrderedDict) -> OrderedDict:
    ''' Convert ETDS to plain network '''
    # Get the hyperparameters of the model according to the checkpoint
    info = get_info(state)
    # Create model objects based on hyperparameters and load checkpoints.
    model1 = ETDS(**info)
    model1.load_state_dict(state)
    model2 = ETDSForInference(**info)

    # for conv_first
    if hasattr(model1.conv_first, 'rep_params'):
        weight, bias = model1.conv_first.rep_params()
    else:
        weight, bias = model1.conv_first.weight.data, model1.conv_first.bias.data
    model2.conv_first.weight.data = torch.cat([weight, model1.conv_residual_first.weight.data])
    model2.conv_first.bias.data = torch.cat([bias, model1.conv_residual_first.bias.data])

    # for backbones
    for backbone_conv, residual_conv, add_residual_conv, layer2 in zip(model1.backbone_convs, model1.residual_convs, model1.add_residual_convs, model2.backbone_convs):
        if hasattr(backbone_conv, 'rep_params'):
            weight, bias = backbone_conv.rep_params()
        else:
            weight, bias = backbone_conv.weight.data, backbone_conv.bias.data
        weight = torch.cat([weight, add_residual_conv.weight.data], dim=1)
        bias = bias + add_residual_conv.bias.data
        residual_weight = torch.cat([torch.zeros(residual_conv.weight.shape[0], weight.shape[1] - residual_conv.weight.shape[1], 3, 3), residual_conv.weight.data], dim=1)
        residual_bias = residual_conv.bias.data
        layer2.weight.data = torch.cat([weight, residual_weight])
        layer2.bias.data = torch.cat([bias, residual_bias])

    # for conv_last and conv_clip
    if hasattr(model1.conv_last, 'rep_params'):
        weight, bias = model1.conv_last.rep_params()
    else:
        weight, bias = model1.conv_last.weight.data, model1.conv_last.bias.data
    model2.conv_last.weight.data[:, :, :, :] = 0
    model2.conv_last.bias.data[:] = 1
    model2.conv_last.weight.data[:weight.shape[0], :, :, :] = -torch.cat([weight, model1.conv_residual_last.weight.data], dim=1)
    model2.conv_last.bias.data[:weight.shape[0]] = 1 - (bias + model1.conv_residual_last.bias.data)
    model2.conv_clip.weight.data[:, :, :, :] = 0
    model2.conv_clip.bias.data[:] = 1
    for i in range(model2.conv_clip.weight.shape[0]):
        model2.conv_clip.weight.data[i, i , 0, 0] = -1
        model2.conv_clip.bias.data[i] = 1

    # Verify that the network output before and after the transformation is the same (due to floating point numbers, etc., it cannot be exactly the same).
    assert model_allclose(model1, model2, info['num_in_ch'])

    return model2.state_dict()


def main():
    ''' main '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='ETDS model state path.')
    parser.add_argument('--output', type=str, required=True, help='ETDS for inference output path.')
    args = parser.parse_args()

    if os.path.split(args.output)[0] != '':
        os.makedirs(os.path.split(args.output)[0], exist_ok=True)

    state = torch.load(args.input)
    converted_state = convert(state)
    torch.save(converted_state, args.output)


if __name__ == '__main__':
    main()
