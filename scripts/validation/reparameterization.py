''' validation reparameterization '''
import random
import torch
from torch.nn import functional as F
from torch.nn.init import kaiming_normal_


def reparameterization(W1, b1, W2, b2):
    ''' W1 * (W2 * x + b2) + b1
    In general, it can be simply implemented through torch.einsum, and the general form is provided here.'''
    # get shape
    c1, c2, k1, k2 = W1.shape
    assert len(b1.shape) == 1 and b1.shape[0] == c1
    c2_, c3, k3, k4 = W2.shape
    assert len(b2.shape) == 1 and b2.shape[0] == c2 == c2_

    # calc (The equivalent form described in the appendix, using matrix multiplication, makes it faster to run)
    W3 = torch.zeros(c1, c3, k1 + k3 - 1, k2 + k4 - 1)
    for k1_ in range(k1):
        for k2_ in range(k2):
            for k3_ in range(k3):
                for k4_ in range(k4):
                    W3[:, :, k1_ + k3_, k2_ + k4_] += W1[:, :, k1_, k2_] @ W2[:, :, k3_, k4_]
    b3 = torch.einsum('oikl,i->o', [W1, b2]) + b1

    return W3, b3


def test_reparameterization(c1, c2, c3, k1, k2, k3, k4, sucess_info=None):
    ''' test reparameterization '''
    W1, b1 = kaiming_normal_(torch.zeros(c1, c2, k1, k2), mode='fan_in'), torch.rand(c1)
    W2, b2 = kaiming_normal_(torch.zeros(c2, c3, k3, k4), mode='fan_in'), torch.rand(c2)
    # reparameterization
    W3, b3 = reparameterization(W1, b1, W2, b2)

    input = torch.rand(1, c3, 48, 48)
    out1 = F.conv2d(F.conv2d(input, W2, b2), W1, b1)
    out2 = F.conv2d(input, W3, b3)
    assert abs(out1 - out2).max().item() < 1/2550, abs(out1 - out2).max().item()
    if sucess_info:
        print(sucess_info)


def main():
    ''' main '''
    # fixed
    test_reparameterization(32, 32, 32, 1, 1, 3, 3, 'fixed 1 passed')
    test_reparameterization(32, 32, 32, 3, 3, 1, 1, 'fixed 2 passed')
    test_reparameterization(32, 32, 32, 3, 1, 1, 3, 'fixed 3 passed')
    test_reparameterization(32, 32, 32, 1, 3, 3, 1, 'fixed 3 passed')

    # random
    for i in range(100):
        test_reparameterization(random.randint(1, 64), random.randint(1, 64), random.randint(1, 64),
                                random.randint(0, 7) * 2 + 1, random.randint(0, 7) * 2 + 1,
                                random.randint(0, 7) * 2 + 1, random.randint(0, 7) * 2 + 1, f'random {i} passed')

if __name__ == '__main__':
    main()
