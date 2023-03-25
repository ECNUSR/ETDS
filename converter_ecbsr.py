''' converter '''
import math
import argparse
import os
from typing import OrderedDict
import torch
from core.archs.ir import ECBSR, ECBSRT


def get_info(state: OrderedDict) -> dict:
    ''' Obtain information such as the number of layers and channels of the model according to the checkpoint file. '''
    # num_in_ch, num_out_ch, upscale, num_block, num_feat,
    num_in_ch = num_out_ch = state['backbone.0.conv3x3.weight'].shape[1]
    num_feat = state['backbone.0.conv3x3.weight'].shape[0]
    num_block = (len([name for name in state if 'backbone.' in name]) + 1) // 22 - 2
    upscale = round(math.sqrt(state[[name for name in state if 'conv3x3.weight' in name][-1]].shape[0] // num_out_ch))
    return {
        'num_in_ch': num_in_ch,
        'num_out_ch': num_out_ch,
        'num_feat': num_feat,
        'num_block': num_block,
        'upscale': upscale,
        'act_type': 'prelu',        # assert act_type = prelu
    }


def model_allclose(model1: ECBSR, model2: ECBSRT, num_in_ch: int) -> bool:
    ''' Determine whether the output of the model is similar '''
    # Randomly generate input
    input = torch.rand(4, num_in_ch, 64, 64)
    out1 = torch.clamp(model1(input), min=0, max=1)
    out2 = torch.clamp(model2(input), min=0, max=1)
    return torch.max((out1 - out2).abs()).item() * 255 < 0.01


def get_reparameterization_state(state: OrderedDict):
    ''' get reparameterization state '''
    info = get_info(state)
    model1 = ECBSR(**info)
    model1.load_state_dict(state)
    model2 = ECBSRT(**info)

    # reparameterization for backbones
    for layer1, layer2 in zip(model1.backbone, model2.backbone):
        weight, bias = layer1.rep_params()
        layer2[0].weight.data = weight
        layer2[0].bias.data = bias
        if hasattr(layer1.act, 'weight'):
            layer2[1].weight.data = layer1.act.weight.data

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
    reparameterization_state = get_reparameterization_state(state)
    torch.save(reparameterization_state, args.output)


if __name__ == '__main__':
    main()
