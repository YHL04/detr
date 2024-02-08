

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """
    Compute sinusoid encoding from original transformer paper.

    Parameters:
        d_model (int): dimension of model
        max_len (int): max length of transformer
    """
    def __init__(self, d_model, max_len):
        super(PositionalEncoding, self).__init__()

        self.encoding = torch.zeros(max_len, d_model)
        self.encoding.requires_grad = False

        pos = torch.arange(0, max_len)
        pos = pos.float().unsqueeze(dim=1)

        _2i = torch.arange(0, d_model, step=2).float()

        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))

    def forward(self, x):
        """Obtain positional encoding according to input size"""
        batch_size, seq_len, d_model = x.size()
        return self.encoding[:seq_len, :].to(x.device)


class LearnedPositionalEncoding(nn.Module):
    """
    Learned Positional Embedding

    Parameters:
        d_model (int): Dimension of model
        max_len (int): Max length of transformer
    """

    def __init__(self, d_model, max_len):
        super(LearnedPositionalEncoding, self).__init__()
        self.encoding = nn.Parameter(torch.randn(max_len, d_model),
                                     requires_grad=True)

    def forward(self, x):
        """Return learned positional encoding according to input shape"""
        batch_size, seq_len, d_model = x.size()
        return self.encoding[:seq_len, :]

