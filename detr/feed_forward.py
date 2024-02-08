

import torch.nn as nn


class FeedForward(nn.Module):
    """
    A simple feed forward network to be used in transformer layers.

    Architecture:
        Sequential(
            LayerNorm(dim)
            Linear(dim, inner_dim)
            GELU()
            Linear(inner_dim, dim)
        )

    Args:
        dim (int): The dimension of the input and output
        inner_dim (int): The dimension of the hidden layer
    """

    def __init__(self, dim, inner_dim):
        super(FeedForward, self).__init__()

        self.ff = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU(),
            nn.Linear(inner_dim, dim)
        )

    def forward(self, x):
        return self.ff(x)


class PredictionFeedForward(nn.Module):
    """
    Bounding box prediction MLP for DETR coordinates output
    """
    def __init__(self, dim):
        super(PredictionFeedForward, self).__init__()

        self.ff = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, 4)
        )

    def forward(self, x):
        return self.ff(x)

