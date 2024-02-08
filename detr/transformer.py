

import torch
import torch.nn as nn


from .feed_forward import FeedForward
from .pos_emb import PositionalEncoding
from .attention import Attention


class Encoder(nn.Module):
    """
    Encoder

    Taking in (flattened resnet representation + positional encoding) as input
    """

    def __init__(self, vocab_size):
        super(Encoder, self).__init__()
        self.transformer = Transformer(
            vocab_size=vocab_size,
            max_len=512,
            n_layers=4,
            d_model=768,
            n_head=8,
            p=0.1,
            device="cuda",
        )

    def forward(self, x):
        return self.transformer(x)


class Decoder(nn.Module):
    """
    Decoder

    Taking in encoder outputs and outputting representations for prediction heads
    """

    def __init__(self, vocab_size):
        super(Decoder, self).__init__()
        self.transformer = Transformer(
            vocab_size=vocab_size,
            max_len=512,
            n_layers=4,
            d_model=768,
            n_head=8,
            p=0.1,
            device="cuda",
        )

    def forward(self, x):
        return self.transformer(x)


class Transformer(nn.Module):
    """
    A standard Transformer module that outputs the unprocessed
    output of the last transformer layer

    Parameters:
        vocab_size (int): Vocabulary size
        max_len (int): Max length
        n_layers (int): Number of layers
        d_model (int): Dimension of transformer
        n_head (int): Number of attention heads
        p (int): Dropout probability

    """

    def __init__(self,
                 vocab_size,
                 max_len=512,
                 n_layers=4,
                 d_model=768,
                 n_head=8,
                 p=0.1,
                 device="cuda",
                 **kwargs
                 ):

        super(Transformer, self).__init__()
        self.n_layers = n_layers
        self.d_model = d_model
        self.device = device

        self.embedding = TransformerEmbedding(vocab_size=vocab_size,
                                              d_model=d_model,
                                              max_len=max_len,
                                              device=device)

        self.layers = nn.ModuleList([AttentionLayer(d_model=d_model,
                                                    ffn_hidden=4 * d_model,
                                                    n_head=n_head,
                                                    p=p)
                                    for _ in range(n_layers)])

    def forward(self, ids, is_causal):
        """
        Computes transformer output

        Parameters:
        ids (Tensor[batch_size, length]): tokens
        state (Tensor[batch_size, state_len, d_model]): recurrent state

        Returns:
        x (Tensor[batch_size, length, d_model]): output
        state (Tensor[batch_size, length, d_model]): next recurrent state

        """
        x = self.embedding(ids)

        for layer in self.layers:
            x = layer(x, is_causal=is_causal)

        return x


class AttentionLayer(nn.Module):
    """
    Class representing a standard transformer layer. This layer includes self-attention,
    normalization, dropout, and a feed-forward network

    Args:
        d_model (int): The dimension of the model
        ffn_hidden (int): The size of the hidden layer in the feed forward network
        n_head (int): The number of attention heads
        p (float): The probability of dropout
    """

    def __init__(self, d_model, ffn_hidden, n_head, p):
        super(AttentionLayer, self).__init__()
        self.attention = Attention(d_model=d_model, n_head=n_head)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(p=p)

        self.ffn = FeedForward(dim=d_model, inner_dim=ffn_hidden)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(p=p)

    def forward(self, x, mask=None, is_causal=False):
        """Compute the output of the transformer layer"""
        _x = x
        x = self.attention(q=x, kv=x, mask=mask, is_causal=is_causal)

        x = self.norm1(x + _x)
        x = self.dropout1(x)

        _x = x
        x = self.ffn(x)

        x = self.norm2(x + _x)
        x = self.dropout2(x)

        return x


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


class TokenEmbedding(nn.Module):
    """
    Token Embedding for transformer
    """

    def __init__(self, vocab_size, d_model):
        super(TokenEmbedding, self).__init__()
        self.emb = nn.Embedding(vocab_size, d_model)

    def forward(self, ids):
        """
        Parameters:
        ids : [batch_size, length]
        token_emb : [batch_size, length, dim]
        """
        token_emb = self.emb(ids)
        return token_emb


class TransformerEmbedding(nn.Module):
    """
    Transformer Embedding, combining positional encoding and token embedding
    """

    def __init__(self, vocab_size, d_model, max_len, device):
        super(TransformerEmbedding, self).__init__()
        self.tok_emb = TokenEmbedding(vocab_size, d_model)
        self.pos_emb = PositionalEncoding(d_model, max_len, device)

    def forward(self, x):
        """
        Returns complete transformer embedding for transformer layers

        Parameters:
        x : [batch_size, length]

        Returns:
        token_emb + pos_emb : [batch_size, length, dim]
        """
        token_emb = self.tok_emb(x)
        pos_emb = self.pos_emb(token_emb)
        return token_emb + pos_emb

