"""
    @file:              Module.py
    @Author:            Alexandre Ayotte
    @Creation Date:     05/07/2020
    @Last modification: 05/07/2020

    @ Reference 1): G. Huang, Y. Sun*, Z. Liu , D. Sedra, K. Q. Weinberger Deep Networks with Stochastic Depth

    The WideResNet Model with manifold mixup and dropout2D.
"""

import torch
from Experimentation.Cifar10_wide_resnet.Model import Module
from Experimentation.Cifar10_wide_resnet.Model.Utils import get_stochastic_depth
import numpy as np


class ResBlock(torch.nn.Module):
    def __init__(self, fmap_in, fmap_out, kernel, subsample=False, drop_rate=0):
        """
        Create a wide residual block
        @Inspired by: https://github.com/vikasverma1077/manifold_mixup

        :param fmap_in: Number of input feature maps
        :param fmap_out: Number of output feature maps
        :param kernel: Kernel size as integer (Example: 3.  For a 3x3 kernel)
        :param subsample: If we want to subsample the image.
        :param drop_rate: The hyperparameter of the Dropout2D module.
        """

        torch.nn.Module.__init__(self)

        self.bn1 = torch.nn.BatchNorm2d(fmap_in)
        self.activation1 = torch.nn.ReLU()

        res_layer = [
            torch.nn.Conv2d(fmap_in, fmap_out, kernel_size=kernel, stride=(2 if subsample else 1),
                            padding=1, bias=True)]

        if drop_rate > 0:
            res_layer.extend([torch.nn.Dropout2d(drop_rate)])

        res_layer.extend([
            torch.nn.BatchNorm2d(fmap_out),
            torch.nn.ReLU(),
            torch.nn.Conv2d(fmap_out, fmap_out, kernel_size=kernel, stride=1,
                            padding=1, bias=True)
        ])

        self.residual_layer = torch.nn.Sequential(*res_layer)

        # If subsample is True
        self.subsample = subsample or (fmap_in != fmap_out)
        if subsample or fmap_in != fmap_out:
            self.sub_conv = torch.nn.Conv2d(fmap_in, fmap_out, kernel_size=1, stride=(2 if subsample else 1), bias=True)

    def forward(self, x):
        """
        Define the forward pass of the Residual layer

        :param x: Input tensor of the convolutional layer
        :return: Output tensor of the residual block
        """

        out = self.bn1(x)
        out = self.activation1(out)

        if self.subsample:
            shortcut = self.sub_conv(out)
        else:
            shortcut = x

        out = self.residual_layer(out) + shortcut

        return out


class WideResNet(torch.nn.Module):
    def __init__(self, nb_layer=40, widen_factor=4, task="CIFAR10", drop_rate=0, mixup=None):
        """
        The WideResNet architecture

        :param nb_layer: Number of layer of the model. (Option: 28, 40)
        :param widen_factor parameters of the wide_resnet.
        :param task: The dataset on which the model will be train. (Option: CIFAR, STL10 or ImageNet)
        :param drop_rate: The maximum dropout rate that will be used following instruction gived by (Ref 1).
        :param mixup: A list of 4 element that the define the hyper-parameters of the 4 mixup modules.
                      If the hyper-parameter is 0, then the corresponding mixup module will not create and then not add
                      to the mixup_index list.
        """

        super().__init__()

        self.conv_layers = None
        self.fc_layer = None
        self.mixup_index = []
        self.num_flat_features = None

        if mixup is None:
            mixup = [0, 0, 0, 0]  # Input, level1, level2, level3

        if task == "CIFAR10":
            self.num_classes = 10
        elif task == "CIFAR100":
            self.num_classes = 100
        elif task == "Imagenet":
            self.num_classes = 1000

        self.__build_model(nb_layer, widen_factor, task, drop_rate, mixup)

    def __build_model(self, nb_layer, widen_factor, task, drop_rate, mixup):
        """
        Create the architecture of the model following the givens parameters.

        :param nb_layer: Number of layer of the model. (Option: 28, 40)
        :param widen_factor parameters of the wide_resnet.
        :param task: The dataset on which the model will be train. (Option: CIFAR, STL10 or ImageNet)
        :param drop_rate: The dropout rate of the Dropout2D module.
        :param mixup: A list of 5 element that the define the hyper-parameters of the 5 mixup modules.
                      If the hyper-parameter is 0, then the corresponding mixup module will not create and then not add
                      to the mixup_index list.
        """

        layers = {22: [3, 3, 3], 28: [4, 4, 4], 40: [6, 6, 6]}
        config = layers[nb_layer]
        conv_list = []

        # ------------------------------------------------------------------------------------------
        #                                      INPUT PART
        # ------------------------------------------------------------------------------------------

        if mixup[0] > 0:
            conv_list.extend([Module.Mixup(mixup[0])])
            self.mixup_index.extend([0])

        if task == "CIFAR10":
            out_channels = 16
            ker_size = 3
            num_classes = 10
        elif task == "CIFAR100":
            out_channels = 16
            ker_size = 3
            num_classes = 100
        else:
            out_channels = 16
            ker_size = 7
            num_classes = 1000

        conv_list.extend([torch.nn.Conv2d(3, out_channels, kernel_size=ker_size, padding=1)])

        # ------------------------------------------------------------------------------------------
        #                                      RESIDUAL PART
        # ------------------------------------------------------------------------------------------
        f_in = f_out = out_channels
        drop_rate_list = get_stochastic_depth(np.sum(layers[nb_layer]), max_drop_rate=drop_rate)
        for it in range(len(config)):
            conv_list.extend([ResBlock(f_in * widen_factor if it > 0 else f_in,
                                       f_out * widen_factor,
                                       kernel=3,
                                       subsample=(it != 0),
                                       drop_rate=drop_rate_list.pop(0))])

            f_in = f_out

            for _ in range(config[it] - 1):
                conv_list.extend([ResBlock(f_in * widen_factor,
                                           f_out * widen_factor,
                                           kernel=3,
                                           drop_rate=drop_rate_list.pop(0))])

            # We add the mixup module.
            if mixup[it + 1]:
                self.mixup_index.append(len(conv_list))
                conv_list.extend([Module.Mixup(mixup[it + 1])])

            if it < len(config) - 1:
                f_out *= 2

        conv_list.extend([torch.nn.BatchNorm2d(f_out*widen_factor, momentum=0.9),
                          torch.nn.ReLU(),
                          torch.nn.AvgPool2d(8)]
                         )

        self.conv_layers = torch.nn.Sequential(*conv_list)

        # ------------------------------------------------------------------------------------------
        #                                   FULLY CONNECTED PART
        # ------------------------------------------------------------------------------------------
        self.num_flat_features = f_out*widen_factor

        self.fc_layer = torch.nn.Sequential(torch.nn.Linear(f_out*widen_factor, num_classes))

    def forward(self, x):
        """
        Define the forward pass of the neural network

        :param x: Input tensor of size BxD where B is the Batch size and D is the features dimension
        :return: Output tensor of size num_class x 1.
        """

        conv_out = self.conv_layers(x)
        output = self.fc_layer(conv_out.view(-1, self.num_flat_features))
        return output

    def configure_mixup_module(self, batch_size=150, device="cuda:0"):
        """
        Set up the batch size and device parameters of all mixup modules present in the model.

        :param batch_size: The batch_size that will be used to train the model.
        :param device: The device on which the model will be train.
        """

        for i in self.mixup_index:
            self.conv_layers[i].batch_size = batch_size
            self.conv_layers[i].device = device

    def restore(self, checkpoint_path):
        """
        Restore the weight from the last checkpoint saved during training

        :param checkpoint_path:
        """

        if checkpoint_path is not None:
            checkpoint = torch.load(checkpoint_path)
            self.load_state_dict(checkpoint['model_state_dict'])
            return checkpoint['epoch'], checkpoint['loss'], checkpoint['accuracy']
