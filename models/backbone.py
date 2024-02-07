

import torch
import torch.nn as nn


class SinusoidalPosEmb(nn.Module):
    """
    Sinusoidal Positional Embedding


    """

    def __init__(self):
        super(SinusoidalPosEmb, self).__init__()
        pass

    def forward(self, x):
        return x


class ResNet(nn.Module):
    """
    Resnet


    """

    def __init__(self):
        super(Backbone, self).__init__()
        self.resnet = None

    def forward(self, x):
        return self.resnet(x)


class Backbone(nn.Module):
    """
    Resnet backbone


    """

    def __init__(self):
        super(Backbone, self).__init__()
        self.cnn = ResNet()
        self.pos_emb = SinusoidalPosEmb()

    def forward(self, x):
        return self.cnn(x) + self.pos_emb(x)

