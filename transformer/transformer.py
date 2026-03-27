import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

## TODOs
# Triangle mask
# Cross attention
# Decoding
# Training loop
# Physics mask

class TrafficEncoding(nn.Module):
    def __init__(self, num_features, d_model):
        """
        Inputs:
        d_model: The dimension of the embeddings. 
        T: Number of timesteps in input
        n_cells: Number of cells in input
        """
        super(TrafficEncoding, self).__init__()

        self.embed = nn.Linear(in_features=num_features, out_features=d_model)

    def spatial_encoding(self, x, d_model):
        n_cells = x.shape[1]
        cell_indices = torch.arange(n_cells).unsqueeze(1) # (N, 1)
        i = torch.arange(d_model).unsqueeze(0) # (1, d_model)

        denoms = 1 / torch.pow(torch.full((1, d_model), 10000), (2*i / d_model))
        angles = cell_indices * denoms
        
        encodings = torch.zeros(n_cells, d_model)
        encodings[:, 0::2] = torch.sin(angles[:, 0::2])
        encodings[:, 1::2] = torch.cos(angles[:, 1::2])

        return encodings
    
    def temporal_encoding(self, x, d_model):
        T = x.shape[0]
        time_steps = torch.arange(T).unsqueeze(1) # (T, 1)
        i = torch.arange(d_model).unsqueeze(0) # (1, d_model)

        denoms = 1 / torch.pow(torch.full((1, d_model), 10000), (0 / d_model))
        angles = time_steps * denoms

        encodings = torch.zeros(T, d_model)
        encodings[:, 0::2] = torch.sin(angles[:, 0::2])
        encodings[:, 1::2] = torch.cos(angles[:, 1::2])

        return encodings

    def forward(self, x):
        """
        Embeds x and adds spatial and temporal encoding to the model input x.
        """
        x = self.embed(x) # x: (T, n_cells, d_model)

        p_space = self.spatial_encoding(x, d_model).unsqueeze(0)   # (1, n_cells, d_model)
        p_time = self.temporal_encoding(x, d_model).unsqueeze(1)    # (T, 1, d_model)

        embeddings = x + p_space + p_time
        
        return embeddings

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        """
        Inputs:
        d_model: The dimension of the embeddings.
        num_heads: The number of attention heads to use.
        """
        super(MultiHeadAttention, self).__init__()

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def split_heads(self, x):
        """
        Reshapes Q, K, V into multiple heads.
        """
        T, n_cells, _ = x.shape
        return x.view(T, n_cells, self.num_heads, self.d_k).permute(0, 2, 1, 3) # (T, num_heads, n_cells, d_k)

    def compute_attention(self, Q, K, V):
        """
        Returns attention between Q, K, and V.
        """
        raw = (Q @ K.transpose(-2, -1)) / np.sqrt(d_k) # Q: (T, num_heads, n_cells, d_k), K.T: (T, num_heads, d_k, n_cells) -> raw: (T, num_heads, n_cells, n_cells)
        # physics mask + triangular mask
        weights = F.softmax(raw, dim=-1)
        attention = weights @ V # attention: (T, num_heads, n_cells, d_k)
        return attention

    def combine_heads(self, x):
        """
        Concatenates the outputs of each attention head into a single output.
        """
        T, _, n_cells, _ = x.size()
        return x.permute(0, 2, 1, 3).contiguous().view(T, n_cells, self.d_model)

    def forward(self, x):
        Q = self.W_q(x) # x: (T, n_cells, d_model), Q: (T, n_cells, d_model)
        K = self.W_k(x)
        V = self.W_v(x)

        Q = self.split_heads(Q) # Q: (T, num_heads, n_cells, d_k)
        K = self.split_heads(K)
        V = self.split_heads(V)

        output = self.compute_attention(Q, K, V) # output: (T, num_heads, n_cells, d_k)
        output = self.combine_heads(output) # output: (T, n_cells, d_model)
        output = self.W_o(output)

        return output

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        """
        Inputs:
        d_model: The dimension of the embeddings.
        d_ff: Hidden dimension size for the feed-forward network.
        """
        super(FeedForward, self).__init__()

        self.fc1 = nn.Linear(d_model, d_ff)
        self.ReLU = nn.ReLU()
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.fc1(x)
        x = self.ReLU(x)
        x = self.fc2(x)

        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, p):
        """
        Inputs:
        d_model: The dimension of the embeddings.
        num_heads: Number of heads to use for mult-head attention.
        d_ff: Hidden dimension size for the feed-forward network.
        p: Dropout probability.
        """
        super(DecoderLayer, self).__init__()

        self.spatial_attn = self.MultiHeadAttention(d_model, num_heads)
        self.temporal_attn = self.MultiHeadAttention(d_model, num_heads)
        self.spatial_attn_norm = nn.LayerNorm(d_model)
        self.temporal_attn_norm = nn.LayerNorm(d_model)
        self.feed_forward = self.FeedForward(d_model, d_ff)
        self.ff_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p)

    def forward(self, x):
        # Spatial Attention
        norm_x = self.spatial_attn_norm(x)
        attn_output = self.spatial_attn(norm_x)
        attn_output = self.dropout(attn_output)
        x = x + attn_output  # x: (T, n_cells, d_model)

        # Temporal Attention
        x = x.permute(1, 0, 2).contiguous() # x: (n_cells, T, d_model)
        norm_x = self.temporal_attn_norm(x)
        attn_output = self.temporal_attn(norm_x)
        attn_output = self.dropout(attn_output)
        x = x + attn_output
        x = x.permute(1, 0, 2).contiguous()

        # Cross-Attention

        # Feedforward
        norm_x = self.ff_norm(x)
        ff_output = self.feed_forward(norm_x)
        ff_output = self.dropout(ff_output)
        x = x + ff_output

        return x

class Transformer(nn.Module):
    def __init__(self, num_features, d_model, num_heads, num_layers, d_ff, num_features, p):
        """
        Inputs:
        num_classes: Number of classes in the classification output.
        d_model: The dimension of the embeddings.
        num_heads: Number of heads to use for mult-head attention.
        num_layers: Number of encoder layers.
        d_ff: Hidden dimension size for the feed-forward network.
        max_seq_length: Maximum sequence length accepted by the transformer.
        p: Dropout probability.
        """
        super(Transformer, self).__init__()

        self.encoding = self.TrafficEncoding(num_features, d_model)
        self.dropout = nn.Dropout(p)
        self.decoder_layers = nn.ModuleList([self.DecoderLayer(d_model, num_heads, d_ff, p) for _ in range(num_layers)])

    def forward(self, x):
        x = self.encoding(x)
        x = self.dropout(x)
        x = self.decoder_layers(x)

        # Decode

        return x


def main():
    # Training Loop
    pass


if __name__ == "__main__":
    main("prompt")