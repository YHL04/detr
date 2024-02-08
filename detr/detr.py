

import torch
import torch.nn as nn

from backbone import Backbone
from transformer import Encoder, Decoder
from feed_forward import FeedForward


class DETR(nn.Module):
    """
    1. Backbone
        CNN (image to 2d representations)
        positional encoding

    2. Encoder
        Transformer Encoder to learn representations

    3. Decoder
        Transformer Decoder to output object representations

    4. Prediction Heads
        FFN to transform representations into set predictions


    Parameters:
        N (int): Decoder query slots, or maximum number of objects that can be detected at once

    """

    def __init__(self, N=100):
        super(DETR, self).__init__()

        # hyper-parameters
        self.N = N

        # modules
        self.backbone = Backbone()
        self.encoder = Encoder()
        self.decoder = Decoder(N)
        self.pred_heads = FeedForward()

    def forward(self, x):
        """
        Diagram from paper (pg 7)
        https://arxiv.org/pdf/2005.12872.pdf



        """
        x = self.backbone(x)
        print(x.shape)
        x = self.encoder(x)
        print(x.shape)
        x = self.decoder(x)
        print(x.shape)
        x = self.pred_heads(x)
        print(x.shape)

        return x