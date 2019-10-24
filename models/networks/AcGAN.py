# -*- coding: utf-8 -*-
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):
    """Generator. Encoder-Decoder Architecture."""
    def __init__(self, conv_dim=64, c_dim=5, repeat_num=6):
        super(Generator, self).__init__()
        self._name = 'generator_wgan'

        layers = []
        layers.append(nn.Conv2d(3 + c_dim, conv_dim, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(nn.InstanceNorm2d(conv_dim, affine=True))
        layers.append(nn.ReLU(inplace=True))

        # Down-Sampling
        curr_dim = conv_dim
        for i in range(2):
            layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(curr_dim*2, affine=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim * 2

        # Bottleneck
        for i in range(repeat_num):
            layers.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim))

        # Up-Sampling
        for i in range(2):
            layers.append(nn.ConvTranspose2d(curr_dim, curr_dim//2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(curr_dim//2, affine=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim // 2

        self.main = nn.Sequential(*layers)

        layers = []
        layers.append(nn.Conv2d(curr_dim, 3, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(nn.Tanh())
        self.img_reg = nn.Sequential(*layers)

        layers = []
        layers.append(nn.Conv2d(curr_dim, 1, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(nn.Sigmoid())
        self.attention_reg = nn.Sequential(*layers)

        # # initial networks
        # for m in self.modules():
        #     if isinstance(m, (nn.Conv2d, nn.Conv1d)):
        #         # nn.init.xavier_uniform_(m.weight.data)
        #         m.weight.data.normal_(0, 0.02)
        #     elif isinstance(m, nn.Linear):
        #         m.weight.data.normal_(0, 0.02)
        #         nn.init.constant_(m.bias.data, 0.0)
        #     elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
        #         nn.init.normal_(m.weight.data, 1.0, 0.02)
        #         nn.init.constant_(m.bias.data, 0.0)


    def forward(self, x, c):
        # replicate spatially and concatenate domain information
        c = c.unsqueeze(2).unsqueeze(3)
        c = c.expand(c.size(0), c.size(1), x.size(2), x.size(3))
        x = torch.cat([x, c], dim=1)
        features = self.main(x)
        return self.img_reg(features), self.attention_reg(features)

class ResidualBlock(nn.Module):
    """Residual Block."""
    def __init__(self, dim_in, dim_out):
        super(ResidualBlock, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True)
            )

    def forward(self, x):
        return x + self.main(x)


class Discriminator(nn.Module):
    """Discriminator. PatchGAN."""
    def __init__(self, c_dim=5, image_size=224, conv_dim=64, repeat_num=7, is_ordinal_reg=False):
        super(Discriminator, self).__init__()
        self._name = 'discriminator_wgan'

        layers = []
        layers.append(nn.Conv2d(3, conv_dim, kernel_size=4, stride=2, padding=1))
        # layers.append(nn.InstanceNorm2d(conv_dim, affine=True)) # add BN
        layers.append(nn.LeakyReLU(0.01, inplace=True))

        curr_dim = conv_dim
        for i in range(1, repeat_num):
            layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1))
            # layers.append(nn.InstanceNorm2d(curr_dim*2, affine=True)) # add BN
            layers.append(nn.LeakyReLU(0.01, inplace=True))
            curr_dim = curr_dim * 2

        k_size = int(image_size / np.power(2, repeat_num))
        self.main = nn.Sequential(*layers)
        self.prob_layer = nn.Conv2d(curr_dim, 1, kernel_size=3, stride=1, padding=1, bias=False)
        if is_ordinal_reg:
            self.cond_layer = nn.Conv2d(curr_dim, c_dim - 1, kernel_size=k_size, bias=False)
        else:
            self.cond_layer = nn.Conv2d(curr_dim, c_dim, kernel_size=k_size, bias=False)

    def forward(self, x):
        h = self.main(x)
        out_real = self.prob_layer(h)
        out_aux = self.cond_layer(h)
        return out_real.squeeze(), F.softmax(out_aux.squeeze(), dim=1)

class PatchDiscriminator(nn.Module):
    """Discriminator. PatchGAN."""
    def __init__(self, c_dim=5, image_size=224, conv_dim=64, repeat_num=6):
        super(PatchDiscriminator, self).__init__()
        self._name = 'discriminator_wgan'

        layers = []
        layers.append(nn.Conv2d(3, conv_dim, kernel_size=4, stride=2, padding=1))
        layers.append(nn.BatchNorm2d(conv_dim))
        layers.append(nn.LeakyReLU(0.2, inplace=True))

        curr_dim = conv_dim
        for i in range(1, repeat_num):
            layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1))
            layers.append(nn.BatchNorm2d(curr_dim*2))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            curr_dim = curr_dim * 2

        k_size = int(image_size / np.power(2, repeat_num))
        self.main = nn.Sequential(*layers)
        self.prob_layer = nn.Conv2d(curr_dim, curr_dim, kernel_size=3, stride=1, padding=1, bias=False)
        self.cond_layer = nn.Conv2d(curr_dim, c_dim, kernel_size=k_size, bias=False)

    def forward(self, x):
        h = self.main(x)
        out_real = self.prob_layer(h)
        out_aux = self.cond_layer(h)
        return out_real.squeeze(), F.softmax(out_aux.squeeze(), dim=1)


if __name__ == '__main__':
    x = torch.randn(2, 3, 224, 224)
    D = PatchDiscriminator()
    p, o = D(x)
    print(D)
    print(p.size(), o.size())
