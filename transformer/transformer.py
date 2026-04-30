import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import math

## TODOs
# torch.nn.functional.scaled_dot_product_attention (Gemini)
# Validation
# Decoding - converting from nodes/links
# make autoregressive
# Physics mask
# Hyperparameters

# Decoder-only - autoregressive prediction
# or not autoregressive? predict interval of future from interval of past?

class TrafficEncoding(nn.Module):
    def __init__(self, num_features, d_model):
        """
        Inputs:
        d_model: The dimension of the embeddings. 
        T: Number of timesteps in input
        n_cells: Number of cells in input
        """
        super(TrafficEncoding, self).__init__()

        self.d_model = d_model
        self.embed = nn.Linear(in_features=num_features, out_features=d_model)

    def spatial_encoding(self, x, d_model):
        n_cells = x.shape[1]
        cell_indices = torch.arange(n_cells).unsqueeze(1) # (N, 1)
        i = torch.arange(self.d_model).unsqueeze(0) # (1, d_model)

        denoms = 1 / torch.pow(torch.full((1, self.d_model), 10000), (2*i / self.d_model))
        angles = cell_indices * denoms
        
        encodings = torch.zeros(n_cells, self.d_model)
        encodings[:, 0::2] = torch.sin(angles[:, 0::2])
        encodings[:, 1::2] = torch.cos(angles[:, 1::2])

        return encodings
    
    def temporal_encoding(self, x, d_model):
        self.d_model = d_model
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

        p_space = self.spatial_encoding(x, self.d_model).unsqueeze(0)   # (1, n_cells, d_model)
        p_time = self.temporal_encoding(x, self.d_model).unsqueeze(1)    # (T, 1, d_model)

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
        print(x.shape)
        T, n_cells, _ = x.shape
        return x.view(T, n_cells, self.num_heads, self.d_k).permute(0, 2, 1, 3) # (T, num_heads, n_cells, d_k)

    def compute_attention(self, Q, K, V, mask=None):
        """
        Returns attention between Q, K, and V.
        """
        raw = (Q @ K.transpose(-2, -1)) / np.sqrt(d_k) # Q: (T, num_heads, n_cells, d_k), K.T: (T, num_heads, d_k, n_cells) -> raw: (T, num_heads, n_cells, n_cells)
        # physics mask + triangular mask
        if mask:
            raw += mask
        weights = F.softmax(raw, dim=-1)
        attention = weights @ V # attention: (T, num_heads, n_cells, d_k)
        return attention

    def combine_heads(self, x):
        """
        Concatenates the outputs of each attention head into a single output.
        """
        T, _, n_cells, _ = x.size()
        return x.permute(0, 2, 1, 3).contiguous().view(T, n_cells, self.d_model)

    def forward(self, x, mask=None):
        Q = self.W_q(x) # x: (T, n_cells, d_model), Q: (T, n_cells, d_model)
        K = self.W_k(x)
        V = self.W_v(x)

        Q = self.split_heads(Q) # Q: (T, num_heads, n_cells, d_k)
        K = self.split_heads(K)
        V = self.split_heads(V)

        output = self.compute_attention(Q, K, V, mask) # output: (T, num_heads, n_cells, d_k)
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

        # Self-Attention
        self.spatial_attn = MultiHeadAttention(d_model, num_heads)
        self.temporal_attn = MultiHeadAttention(d_model, num_heads)
        self.spatial_attn_norm = nn.LayerNorm(d_model)
        self.temporal_attn_norm = nn.LayerNorm(d_model)

        # Feed-Forward
        self.feed_forward = FeedForward(d_model, d_ff)
        self.ff_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p)

    def forward(self, x):
        # Spatial Attention
        norm_x = self.spatial_attn_norm(x)
        attn_output = self.spatial_attn(x)
        attn_output = self.dropout(attn_output)
        x = x + attn_output  # x: (T, n_cells, d_model)

        # Temporal Attention
        x = x.permute(1, 0, 2).contiguous() # x: (n_cells, T, d_model)
        norm_x = self.temporal_attn_norm(x)
        mask_size = x.shape[1]
        self_attn_mask = (1 - np.tri(mask_size)) * -1e10
        attn_output = self.temporal_attn(x, mask=mask)
        attn_output = self.dropout(attn_output)
        x = x + attn_output
        x = x.permute(1, 0, 2).contiguous()

        # Feed-Forward
        norm_x = self.ff_norm(x)
        ff_output = self.feed_forward(norm_x)
        ff_output = self.dropout(ff_output)
        x = x + ff_output

        return x

class Transformer(nn.Module):
    def __init__(self, num_features, d_model, num_heads, num_layers, d_ff, p):
        """
        Inputs:
        num_classes: Number of classes in the classification output.
        d_model: The dimension of the embeddings.
        num_heads: Number of heads to use for mult-head attention.
        num_layers: Number of encoder layers.
        d_ff: Hidden dimension size for the feed-forward network.
        p: Dropout probability.
        """
        super(Transformer, self).__init__()

        self.encoding = TrafficEncoding(num_features, d_model)
        self.dropout = nn.Dropout(p)
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, p) for _ in range(num_layers)])
        self.final_norm = nn.LayerNorm(d_model)
        self.out_projection = nn.Linear(d_model, num_features) # How to get back to features

    def forward(self, x):
        x = self.encoding(x)
        x = self.dropout(x)

        for layer in self.decoder_layers:
            x = layer(x)

        # Decode
        x = self.final_norm(x)
        logits = self.out_projection(x)

        return logits

def compute_loss():
    # todo: loss function -> write question
    pass

def train(model, train_loader, epochs, criterion, optimizer, device):
    # todo: get dataloader

    train_loss_arr = []
    running_loss = 0.0

    for epoch in range(epochs):
        model.train()
        for i, (inputs, labels) in enumerate(train_loader): # labels??
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            pred = model(inputs)
            loss = criterion(pred, labels)
            loss.backward()
            optimizer.step()

        running_loss += loss.item()

        print(
            "epoch:", epoch + 1, 
            "training loss:", running_loss,
        )
    
    return train_loss_arr


def get_static_features(link_filepath):
    links_df = pd.read_csv(link_filepath)

    dt = 5 / 3600  # 5 seconds in hours
    link_type_onehot = pd.get_dummies(links_df["link_type"]).values

    cells = []
    cell_features = []

    for i, row in links_df.iterrows():
        free_speed = row["free_speed"]
        length = row["length"]
        lanes = row["lanes"]
        capacity = row["capacity"]

        cell_length = free_speed * dt
        n_cells = math.ceil(length / cell_length)

        k_jam = 120 * lanes
        k_crit = capacity / free_speed
        wave_speed = capacity / (k_jam - k_crit + 1e-6)
        Q_cell = capacity * dt
        N_max = k_jam * cell_length 

        link_type = link_type_onehot[i]

        for k in range(n_cells):
            cells.append((row["link_id"], k))

            cell_features.append([
                free_speed,
                Q_cell,
                k_jam,
                wave_speed,
                lanes,
                *link_type
            ])

    cell_features = np.array(cell_features)

    return cells, cell_features

def get_dynamic_features(demand_filepath, cells):
    N = len(cells)
    link_to_cells = {}

    for i, (link_id, k) in enumerate(cells):
        link_to_cells.setdefault(link_id, []).append(i)

    demand_df = pd.read_csv(demand_filepath)

    unique_times = sorted(demand_df["time_period"].unique())
    time_to_idx = {t: i for i, t in enumerate(unique_times)}
    demand_df["t_idx"] = demand_df["time_period"].map(time_to_idx)
    T = len(unique_times)

    F_dyn = 4  # density, flow, speed, queue

    dynamic_features = np.zeros((T, N, F_dyn))

    for _, row in demand_df.iterrows():
        t = row["t_idx"]
        link_id = row["link_id"]

        if link_id not in link_to_cells:
            continue

        density = row["density"]
        flow = row["volume"]
        speed = row["RT_speed"]

        k_jam = 120 * row["lanes"]
        queue = k_jam * row["queue_link_distance_in_km"]

        for cell_idx in link_to_cells[link_id]:
            dynamic_features[t, cell_idx, :] = [
                density,
                flow,
                speed,
                queue
            ]

    return dynamic_features, T

def get_training_samples(X, T, T_total):
    inputs = []
    targets = []

    for t in range(T_total - T):
        x = X[t:t+T]   # (T, N, F)
        y = X[t+T]       # (N, F)

        inputs.append(x)
        targets.append(y)

    return inputs, targets

class TrafficDataset(torch.utils.data.Dataset):
    def __init__(self, X, Y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        x = self.X[i]   # (T, N, F)
        y = self.Y[i]

        # remove batch dimension for now
        x = x.squeeze(0)       # (N, F)
        y = y.squeeze(0)       # (N, F)

        return x, y
        # return self.X[i], self.Y[i]

def main():
    # Adjust hyperparameters
    d_model = 300
    num_heads = 4
    num_layers = 4
    d_ff = 1024
    max_seq_length = 40
    dropout = 0.1
    num_features = 12
    transformer_epochs = 10
    lr = 0.0001

    cells, static_features = get_static_features("micro_link.csv")

    dynamic_features, T_total = get_dynamic_features("dynamic_link_performance.csv", cells)
    # print(static_features.shape)
    # print(dynamic_features.shape)

    static_expanded = np.repeat(static_features[np.newaxis, :, :], T_total, axis=0)

    X = np.concatenate([dynamic_features, static_expanded], axis=-1)
    # print(X.shape)

    inputs, targets = get_training_samples(X, 1, T_total)
    # print(inputs[0].shape)
    # print(targets[0].shape)
    inputs = np.array(inputs)
    targets = np.array(targets)

    dataset = TrafficDataset(inputs, targets)
    dataloader = DataLoader(dataset, batch_size=1)

    # # Example usage
    # for features, labels in dataloader:
    #     print(features.shape)
    #     print(labels.shape)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    criterion = nn.MSELoss()
    transformer = Transformer(num_features, d_model, num_heads, num_layers, d_ff, dropout).to(device)
    optimizer = optim.Adam(transformer.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9)
    
    transformer_train_loss = train(transformer, dataloader, transformer_epochs, criterion, optimizer, device)


if __name__ == "__main__":
    main()