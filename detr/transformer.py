

import torch
import torch.nn as nn
import torch.nn.functional as F


from .feed_forward import FeedForward


class Encoder(nn.Module):
    """
    Encoder

    Taking in (flattened resnet representation + positional encoding) as input
    """

    def __init__(self, n_layers, d_model, n_head, p):
        super(Encoder, self).__init__()

        self.layers = nn.ModuleList([EncoderLayer(d_model=d_model,
                                                  ffn_hidden=4 * d_model,
                                                  n_head=n_head,
                                                  p=p)
                                    for _ in range(n_layers)])

    def forward(self, x, pos_emb):
        for layer in self.layers:
            x = layer(x + pos_emb)

        return x


class Decoder(nn.Module):
    """
    Decoder

    Taking in encoder outputs and outputting representations for prediction heads
    """

    def __init__(self, n_layers, d_model, n_head, p):
        super(Decoder, self).__init__()

        self.layers = nn.ModuleList([DecoderLayer(d_model=d_model,
                                                  ffn_hidden=4 * d_model,
                                                  n_head=n_head,
                                                  p=p)
                                    for _ in range(n_layers)])

    def forward(self, obj_queries, kv):
        x = torch.zeros_like(obj_queries, dtype=torch.float32)

        for layer in self.layers:
            x = x + obj_queries
            x = layer(x, kv)

        return x


class EncoderLayer(nn.Module):
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
        super(EncoderLayer, self).__init__()
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


class DecoderLayer(nn.Module):
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
        super(DecoderLayer, self).__init__()
        self.self_attention = Attention(d_model=d_model, n_head=n_head)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(p=p)

        self.cross_attention = Attention(d_model=d_model, n_head=n_head)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(p=p)

        self.ffn = FeedForward(dim=d_model, inner_dim=ffn_hidden)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout3 = nn.Dropout(p=p)

    def forward(self, x, kv, mask=None, is_causal=False):
        """Compute the output of the transformer layer"""
        # Self attention
        _x = x
        x = self.self_attention(q=x, kv=x, mask=mask, is_causal=is_causal)

        x = self.norm1(x + _x)
        x = self.dropout1(x)

        # Cross attention
        _x = x
        x = self.cross_attention(q=x, kv=kv, mask=mask, is_causal=is_causal)

        x = self.norm1(x + _x)
        x = self.dropout1(x)

        # Feed forward
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


class Attention(nn.Module):
    """
    Attention module for Transformer layers.
    Composes of learnable parameters in
    query, key, value and concat linear modules.
    """

    def __init__(self, d_model, n_head):
        super(Attention, self).__init__()
        self.n_head = n_head

        self.w_q = nn.Linear(d_model, d_model, bias=True)
        self.w_k = nn.Linear(d_model, d_model, bias=True)
        self.w_v = nn.Linear(d_model, d_model, bias=True)
        self.w_concat = nn.Linear(d_model, d_model, bias=True)

        # self.w_q = nn.Linear(d_model, d_model, bias=False)
        # self.w_kv = nn.Linear(d_model, 2 * (d_model // n_head), bias=False)
        # self.w_concat = nn.Linear(d_model, d_model, bias=False)

    def forward(self, q, kv, mask=None, is_causal=False):
        """
        Parameters:
        q:     [batch_size, length, d_model]
        kv:    [batch_size, length, d_model]

        Returns:
        out:   [batch_size, length, d_model]
        """
        q, k, v = self.w_q(q), self.w_k(kv), self.w_v(kv)
        q, k, v = self.split(q), self.split(k), self.split(v)

        # q, k, v = self.w_q(q), *self.w_kv(kv).chunk(2, dim=-1)
        # q, k, v = self.split(q), k.unsqueeze(1), v.unsqueeze(1)

        # q = q.to(torch.float16)
        # k = k.to(torch.float16)
        # v = v.to(torch.float16)

        # with torch.backends.cuda.sdp_kernel(
        #         enable_flash=True
        # ):
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=is_causal)

        # out = out.to(torch.float32)

        out = self.concat(out)
        out = self.w_concat(out)

        return out

    def split(self, tensor):
        """
        Split tensor into number of head

        Parameters:
        tensor : [batch_size, length, d_model]

        Returns:
        tensor : [batch_size, head, length, d_tensor]
        """
        batch_size, length, d_model = tensor.shape

        d_tensor = d_model // self.n_head
        tensor = tensor.view(batch_size, length, self.n_head, d_tensor).transpose(1, 2)

        return tensor

    def concat(self, tensor):
        """
        Inverse function of self.split(tensor : torch.Tensor)

        Parameters:
        tensor : [batch_size, head, length, d_tensor]
        Returns:
        tensor : [batch_size, length, d_model]
        """
        batch_size, head, length, d_tensor = tensor.shape
        d_model = head * d_tensor

        tensor = tensor.transpose(1, 2).contiguous().view(batch_size, length, d_model)
        return tensor

