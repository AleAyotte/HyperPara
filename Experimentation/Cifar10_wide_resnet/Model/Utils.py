"""
    @file:              Utils.py
    @Author:            Alexandre Ayotte
    @Creation Date:     20/06/2020
    @Last modification: 02/07/2020

    @ Reference 1): G. Huang, Y. Sun*, Z. Liu , D. Sedra, K. Q. Weinberger Deep Networks with Stochastic Depth

    This file contain some useful functions that will be used in this project.
"""

import torch


def init_weights(m):
    """
    Initialize the weights of the fully connected layer and convolutional layer with Xavier normal initialization
    and Kamming normal initialization respectively.

    :param m: A torch.nn module of the current model. If this module is a layer, then we initialize its weights.
    """

    if type(m) == torch.nn.Linear:
        torch.nn.init.kaiming_normal_(m.weight)
        if not (m.bias is None):
            torch.nn.init.zeros_(m.bias)

    elif type(m) == torch.nn.Conv2d:
        torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity="relu")
        if not (m.bias is None):
            torch.nn.init.zeros_(m.bias)

    elif type(m) == torch.nn.BatchNorm2d:
        torch.nn.init.ones_(m.weight)
        if not (m.bias is None):
            torch.nn.init.zeros_(m.bias)


def to_one_hot(inp, num_classes, device="cuda:0"):
    """
    Transform a logit ground truth to a one hot vector

    :param inp: The input vector to transform as a one hot vector
    :param num_classes: The number of classes in the dataset
    :param device: The device on which the result will be return. (Default="cuda:0", first GPU)
    :return: A one hot vector that represent the ground truth
    """
    y_onehot = torch.FloatTensor(inp.size(0), num_classes)
    y_onehot.zero_()

    y_onehot.scatter_(1, inp.unsqueeze(1).data.cpu(), 1)

    return torch.autograd.Variable(y_onehot.to(device), requires_grad=False)


def get_stochastic_depth(nb_layer, max_drop_rate=0.5):
    """
    Compute the dropout rates that will be used following the instruction in (Ref 1)

    :param nb_layer: Number of dropout module.
    :param max_drop_rate: The drop rate will begin at 0 and will increase to max_drop_rate at the last dropout module.
    :return:
    """

    return [1 - (1 - (max_drop_rate * i / (nb_layer-1))) for i in range(nb_layer)]
