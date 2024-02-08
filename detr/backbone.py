

import torch
import torch.nn as nn


from .pos_emb import PositionalEncoding


class ResNet(nn.Module):
    """
    Pretrained resnet from pytorch


    """

    def __init__(self, resnet):
        super(Backbone, self).__init__()
        self.resnet = torch.hub.load(
            "pytorch/vision:v0.10.0", resnet, pretrained=True
        )

    def forward(self, x):
        return self.resnet(x)


class Backbone(nn.Module):
    """
    Backbone powered by pretrained resnet

    Parameters:
        resnet (string): name of the pretrained weights of resnet in pytorch
    """

    def __init__(self, resnet="resnet18"):
        super(Backbone, self).__init__()
        self.resnet = ResNet(resnet)
        self.pos_emb = PositionalEncoding()

    def forward(self, x):
        return self.resnet(x) + self.pos_emb(x)

