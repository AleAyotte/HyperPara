"""
    @file:              Module.py
    @Author:            Alexandre Ayotte
    @Creation Date:     20/06/2020
    @Last modification: 20/06/2020

    This file contain some general module that will be used to in the construction of Neural Net.
"""

import torch
import numpy as np


class Mixup(torch.nn.Module):
    def __init__(self, beta_params):

        """
        The constructor of a mixup module.
        :param beta_params: One single value for the two parameters of a beta distribution
        """

        torch.nn.Module.__init__(self)
        self.beta = beta_params
        self.lamb = 1
        self.batch_size = 0
        self.device = None
        self.permut = None
        self.enable = False

    def sample(self):

        """
        Sample a point in a beta distribution to prepare the mixup
        :return: The coefficient used to mixup the training features during the next foward pass.
                 The permutation indices that will be used to shuffle the features before the next foward pass.
        """

        if self.beta > 0:
            # We activate the module and we sample a value in his beta distribution
            self.enable = True
            self.lamb = np.random.beta(self.beta, self.beta)
        else:
            self.lamb = 1

        self.permut = torch.randperm(self.batch_size)

        return self.lamb, self.permut

    def get_mix_params(self):
        """
        :return: The constant that will be use to mixup the data for the next iteration and a list of index that
                 represents the permutation used for the mixing process.
        """

        return self.lamb, self.permut

    def forward(self, x):
        """
        Forward pass of the mixup module. This is here were we mix the data.

        :return: Return mixed data if the network is in train mode and the module is enable.
        """

        if self.training and self.enable:
            device = x.get_device()
            lamb = torch.from_numpy(np.array([self.lamb]).astype('float32')).to(self.device)
            lamb = torch.autograd.Variable(lamb)
            return lamb*x + (1 - lamb)*x[self.permut.to(self.device)]
        else:
            return x


class SELayer(torch.nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = torch.nn.AdaptiveAvgPool2d(1)
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(channel, channel // reduction, bias=False),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(channel // reduction, channel, bias=False),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
