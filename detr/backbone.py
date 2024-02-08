

import torch
import torch.nn as nn


class Backbone(nn.Module):
    """
    Pretrained resnet from pytorch as backbone
    Remove the last fully connected layer to get the CNN representations

    """

    def __init__(self, d, resnet="resnet18", C=512):
        super(Backbone, self).__init__()
        self.resnet = torch.hub.load(
            "pytorch/vision:v0.10.0", resnet, pretrained=True
        )
        self.resnet = nn.Sequential(*(list(self.resnet.children())[:-2]))
        self.conv = nn.Conv2d(C, d, kernel_size=1, stride=1)

    def forward(self, x):
        """
        Returns:
            x (torch.tensor): (B, d, H/32, W/32)
        """
        x = self.resnet(x)  # (B, C, H/32, W/32)
        x = self.conv(x)    # (B, d, H/32, W/32)

        return x

