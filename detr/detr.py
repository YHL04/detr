

import torch.nn as nn

from .backbone import Backbone
from .embedding import PositionalEncoding, LearnedPositionalEncoding
from .transformer import Encoder, Decoder
from .feed_forward import PredictionFeedForward


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


    Differences compared to original paper:
        - Positional encoding is computed after transforming into transformer input
          different from paper which computes it in 2d representation right after cnn output

        - Learned Positional Encoding is identical to learned embedding in NLP, different
          from paper which computes embedding as a 2d representation first before transforming it
          to transformer input


    Parameters:
        N (int): Decoder query slots, or maximum number of objects that can be detected at once

    """

    def __init__(self, H, W, num_classes, N=100, d=512, n_layers=4, n_head=8, p=0.1):
        super(DETR, self).__init__()

        # hyper-parameters
        self.H, self.W, self.N = H, W, N
        self.num_classes = num_classes
        self.n_layers = n_layers
        self.n_head = n_head
        self.p = p

        # d_model of transformer is simply H/32 * W/32
        d_model = H//32 * W//32

        # backbone and embeddings
        self.backbone = Backbone(d)
        self.pos_emb = PositionalEncoding(d_model, d)
        self.obj_queries = LearnedPositionalEncoding(d_model, N)

        # transformers
        self.encoder = Encoder(n_layers, d_model, n_head, p)
        self.decoder = Decoder(n_layers, d_model, n_head, p)

        # output layers
        self.class_embed = nn.Linear(d_model, num_classes+1)
        self.bbox_embed = PredictionFeedForward(d_model)

    def forward(self, x):
        """
        Diagram from paper (pg 7)
        https://arxiv.org/pdf/2005.12872.pdf



        """
        # Resnet backbone to output (B, d, H/32, W/32)
        x = self.backbone(x)

        B, d, H1, W1 = x.size()
        x = x.view(B, d, H1 * W1)

        # Encoder takes in x and add pos_emb to each input before layer
        pos_emb = self.pos_emb(x)
        x = self.encoder(x, pos_emb)

        # Decoder takes in learned positional embedding (aka object queries)
        # and use encoder output as kv
        obj_queries = self.obj_queries(x)
        x = self.decoder(obj_queries, x)

        # Final decoder representation passed to class output and box output
        output_class = self.class_embed(x)
        output_coords = self.bbox_embed(x)

        return output_class, output_coords

