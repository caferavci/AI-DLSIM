import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.checkpoint import checkpoint
import torch.optim as optim
import math

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

    def spatial_encoding(self, n_cells, device):
        """
        Spatial encoding: sin and cos functions of cell indices
        """
        cell_indices = torch.arange(n_cells, device=device).unsqueeze(1) # (N, 1)
        i = torch.arange(self.d_model, device=device).unsqueeze(0) # (1, d_model)

        denoms = 1 / torch.pow(100000, (2*(i // 2) / self.d_model))
        angles = cell_indices * denoms
        
        encodings = torch.zeros(n_cells, self.d_model, device=device)
        encodings[:, 0::2] = torch.sin(angles[:, 0::2])
        encodings[:, 1::2] = torch.cos(angles[:, 1::2])

        return encodings
    
    def temporal_encoding(self, T, device):
        """
        Temporal encoding: sin and cos functions of time steps
        """
        time_steps = torch.arange(T, device=device).unsqueeze(1) # (T, 1)
        i = torch.arange(self.d_model, device=device).unsqueeze(0) # (1, d_model)

        denoms = 1 / torch.pow(10000, (2 * (i // 2) / self.d_model))
        angles = time_steps * denoms

        encodings = torch.zeros(T, self.d_model, device=device)
        encodings[:, 0::2] = torch.sin(angles[:, 0::2])
        encodings[:, 1::2] = torch.cos(angles[:, 1::2])

        return encodings

    def forward(self, x):
        """
        Embeds x and adds spatial and temporal encoding to the model input x.
        """
        _, T, n_cells, _ = x.shape
        device = x.device
        x = self.embed(x) # x: (B, T, n_cells, d_model)

        p_space = self.spatial_encoding(n_cells, device).view(1, 1, n_cells, self.d_model)   # (1, 1, n_cells, d_model)
        p_time = self.temporal_encoding(T, device).view(1, T, 1, self.d_model)    # (1, T, 1, d_model)

        embeddings = x + p_space + p_time # (B, T, n_cells, d_model)
        
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
        B, seq_len, _ = x.shape
        return x.view(B, seq_len, self.num_heads, self.d_k).permute(0, 2, 1, 3) # (B, num_heads, seq_len, d_k)

    def compute_attention(self, Q, K, V, mask=None):
        """
        Returns attention between Q, K, and V.
        """
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)  # (B, num_heads, seq_len, seq_len)

        # triangular mask
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        weights = F.softmax(scores, dim=-1)
        attention = weights @ V # attention: (B, num_heads, n_cells/T, d_k)
        return attention

    def combine_heads(self, x):
        """
        Concatenates the outputs of each attention head into a single output.
        """
        B, _, seq_len, _ = x.size()
        return x.permute(0, 2, 1, 3).contiguous().view(B, seq_len, self.d_model)

    def forward(self, x, mask=None):
        Q = self.W_q(x) # x: (B, n_cells/T, d_model), Q: (B, n_cells/T, d_model)
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
    def __init__(self, d_model, num_heads, d_ff, p, spatial_groups=None):
        """
        Inputs:
            d_model: The dimension of the embeddings.
            num_heads: Number of heads to use for mult-head attention.
            d_ff: Hidden dimension size for the feed-forward network.
            p: Dropout probability.
            spatial_groups: List of lists of cell indices for each group.
        """
        super(DecoderLayer, self).__init__()

        # Self-Attention
        self.spatial_attn = MultiHeadAttention(d_model, num_heads)
        self.temporal_attn = MultiHeadAttention(d_model, num_heads)
        self.spatial_attn_norm = nn.LayerNorm(d_model)
        self.temporal_attn_norm = nn.LayerNorm(d_model)
        self.spatial_groups = spatial_groups

        # Feed-Forward
        self.feed_forward = FeedForward(d_model, d_ff)
        self.ff_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p)

    def forward(self, x):
        batch_size, t_steps, n_cells, d_model = x.shape

        # Spatial attention over cells for each timestep
        spatial_x = x.view(batch_size * t_steps, n_cells, d_model)
        norm_x = self.spatial_attn_norm(spatial_x)
        if self.spatial_groups is None:
            # Dense fallback: attends over all N cells.
            attn_output = self.spatial_attn(norm_x)
        else:
            # Sparse-by-construction: run attention independently per group,
            # which avoids allocating one global N x N score matrix.
            attn_output = torch.zeros_like(norm_x)
            for group_indices in self.spatial_groups:
                idx = torch.as_tensor(group_indices, device=norm_x.device, dtype=torch.long)
                group_x = norm_x.index_select(1, idx)
                group_out = self.spatial_attn(group_x)
                attn_output[:, idx, :] = group_out
        attn_output = self.dropout(attn_output)
        spatial_x = spatial_x + attn_output
        x = spatial_x.view(batch_size, t_steps, n_cells, d_model)

        # Temporal attention over timesteps for each cell
        temporal_x = x.permute(0, 2, 1, 3).contiguous().view(batch_size * n_cells, t_steps, d_model)
        norm_x = self.temporal_attn_norm(temporal_x)
        attn_output = self.temporal_attn(norm_x)
        attn_output = self.dropout(attn_output)
        temporal_x = temporal_x + attn_output
        x = temporal_x.view(batch_size, n_cells, t_steps, d_model).permute(0, 2, 1, 3).contiguous()

        # Feed-Forward
        norm_x = self.ff_norm(x)
        ff_output = self.feed_forward(norm_x)
        ff_output = self.dropout(ff_output)
        x = x + ff_output

        return x

class Transformer(nn.Module):
    def __init__(self, num_input_features, num_predict_features, d_model, num_heads, num_layers, d_ff, p, spatial_groups=None, use_checkpoint=True):
        """
        Inputs:
            num_input_features: Feature dim per cell going into the encoder (i.e. static + dynamic).
            num_predict_features: Output dim per cell (i.e. dynamic only).
            d_model: The dimension of the embeddings.
            num_heads: Number of heads to use for mult-head attention.
            num_layers: Number of encoder layers.
            d_ff: Hidden dimension size for the feed-forward network.
            p: Dropout probability.
            spatial_groups: List of lists of cell indices for each group.
            use_checkpoint: Whether to use checkpointing.
        """
        super(Transformer, self).__init__()

        # if num_predict_features is None:
            # num_predict_features = NUM_DYNAMIC_FEATURES

        self.num_input_features = num_input_features
        self.num_predict_features = num_predict_features

        self.encoding = TrafficEncoding(num_input_features, d_model)
        self.dropout = nn.Dropout(p)
        self.use_checkpoint = use_checkpoint
        self.decoder_layers = nn.ModuleList(
            [DecoderLayer(d_model, num_heads, d_ff, p, spatial_groups=spatial_groups) for _ in range(num_layers)]
        )
        self.final_norm = nn.LayerNorm(d_model)
        self.out_projection = nn.Linear(d_model, num_predict_features)

    def forward(self, x):
        x = self.encoding(x)
        x = self.dropout(x)

        for layer in self.decoder_layers:
            if self.training and self.use_checkpoint:
                x = checkpoint(layer, x, use_reentrant=False)
            else:
                x = layer(x)

        x = self.final_norm(x)
        logits = self.out_projection(x)

        return logits

def train(model, train_loader, epochs, criterion, optimizer, device):
    train_loss_arr = []

    for epoch in range(epochs):
        running_loss = 0.0
        model.train()
        for i, (inputs, labels) in enumerate(train_loader): # labels??
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad(set_to_none=True)
            pred = model(inputs)
            if labels.dim() == 3:
                # labels are next-step targets: (B, N, num_predict_features)
                pred = pred[:, -1, :, :]
            loss = criterion(pred, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        
        avg_loss = running_loss / len(train_loader)

        print(
            "epoch:", epoch + 1, 
            "training loss:", avg_loss,
        )
    
    return train_loss_arr

def predict_next_states(model, inputs, device):
    """
    Predict next-step states for a batch.

    inputs: (B, T, N, F_in)
    returns: (B, N, num_predict_features) — by default the dynamic channels only.
    """
    model.eval()
    with torch.no_grad():
        inputs = inputs.to(device)
        pred = model(inputs)          # (B, T, N, num_predict_features)
        next_state_pred = pred[:, -1] # (B, N, num_predict_features)
    return next_state_pred.cpu()


def get_static_features(link_filepath):
    """
    Builds per-cell static traffic attributes

    Input: 
        link_filepath: link.csv
    Output: 
        cells: link id and cell index for each cell
        cell_features: static features for each cell
    """
    links_df = pd.read_csv(link_filepath)

    dt = 5 / 3600  # 5 seconds in hours
    link_type_onehot = pd.get_dummies(links_df["link_type"]).values # 8 values for link type

    cells = []
    cell_features = []

    for i, row in links_df.iterrows():
        free_speed = row["free_speed"]
        length = row["length"]
        lanes = row["lanes"]
        capacity = row["capacity"]

        cell_length = (free_speed * 1000) * dt
        # Get number of cells for each link
        n_cells = math.ceil(length / cell_length)

        k_jam = 120 * lanes
        k_crit = capacity / free_speed
        wave_speed = capacity / (k_jam - k_crit + 1e-6)
        Q_cell = capacity * dt
        N_max = k_jam * cell_length 

        link_type = link_type_onehot[i]

        for k in range(n_cells):
            # Get mapping from link id to cell number
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

def get_dynamic_features(dynamic_filepath, cells):
    """
    Builds time-varying traffic state per cell.

    Supports two CSV layouts:

    * Macronet ``td_link_performance.csv``: hourly ``time_period`` strings like
      ``0600:00.000_0700:00.000``, columns ``speed``, ``density``, ``queue_ratio``.
      Same values are broadcast to every cell on ``link_id``.

    Input:
        dynamic_filepath: path to performance CSV
        cells: list of (link_id, cell_index_on_link)

    Output:
        dynamic_features: array (T, N, F_dyn)
        T: number of timesteps
    """
    N = len(cells)
    link_to_cells = {}

    for i, (link_id, k) in enumerate(cells):
        link_to_cells.setdefault(link_id, []).append(i)

    demand_df = pd.read_csv(dynamic_filepath)

    unique_times = sorted(demand_df["time_period"].astype(str).unique())
    time_to_idx = {t: i for i, t in enumerate(unique_times)}
    demand_df = demand_df.assign(t_idx=demand_df["time_period"].astype(str).map(time_to_idx))
    T = len(unique_times)

    # if "queue_ratio" in demand_df.columns:
    # Macronet hourly link performance — predict speed, density, queue_ratio per cell
    F_dyn = 3
    dynamic_features = np.zeros((T, N, F_dyn), dtype=np.float64)

    for _, row in demand_df.iterrows():
        t = int(row["t_idx"])
        link_id = row["link_id"]

        if link_id not in link_to_cells:
            continue

        speed = float(row["speed"])
        density = float(row["density"])
        queue_ratio = float(row["queue_ratio"])

        vec = np.array([speed, density, queue_ratio], dtype=np.float64)
        for cell_idx in link_to_cells[link_id]:
            dynamic_features[t, cell_idx, :] = vec

    return dynamic_features.astype(np.float32), T

    # # Legacy dynamic_link_performance.csv
    # F_dyn = 4  # density, flow, speed, queue (derived)
    # dynamic_features = np.zeros((T, N, F_dyn), dtype=np.float32)

    # for _, row in demand_df.iterrows():
    #     t = int(row["t_idx"])
    #     link_id = row["link_id"]

    #     if link_id not in link_to_cells:
    #         continue

    #     density = row["density"]
    #     flow = row["volume"]
    #     speed = row["speed"]

    #     k_jam = 120 * row["lanes"]
    #     queue = k_jam * row["queue_link_distance_in_km"]

    #     for cell_idx in link_to_cells[link_id]:
    #         dynamic_features[t, cell_idx, :] = [
    #             density,
    #             flow,
    #             speed,
                # queue
    #         ]

    # return dynamic_features, T

def build_spatial_groups(cells, neighbor_window=3):
    """
    Build sparse spatial groups for attention.

    Each group is a contiguous chunk of cells on the same link, with
    width (2 * neighbor_window + 1). Attention is run per chunk only.
    This approximates adjacency-masked attention while avoiding dense N^2 memory.
    """
    link_to_cells = {}
    for idx, (link_id, _) in enumerate(cells):
        link_to_cells.setdefault(link_id, []).append(idx)

    groups = []
    chunk_size = 2 * neighbor_window + 1
    for cell_indices in link_to_cells.values():
        for start in range(0, len(cell_indices), chunk_size):
            groups.append(cell_indices[start:start + chunk_size])
    return groups

def get_training_samples(X, lookback_window, T):
    """
    Converts time series into supervised learning samples

    Input:
        X: (T, N, F)
        lookback_window: number of timesteps to predict
        T: total number of timesteps
    Output:
        inputs: (T - lookback_window, lookback_window, N, F)
        targets: (T - lookback_window, N, F)
    """
    inputs = []
    targets = []

    for t in range(T - lookback_window):
        x = X[t:t+lookback_window]   # (lookback_window, N, F)
        y = X[t+lookback_window]       # (N, F)

        inputs.append(x)
        targets.append(y)

    return inputs, targets

class TrafficDataset(torch.utils.data.Dataset):
    """
    PyTorch wrapper around dataset to prepare for DataLoader
    """
    def __init__(self, X, Y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        x = self.X[i]   # (T, N, F)
        y = self.Y[i]

        return x, y

def main():
    # Hyperparameters
    d_model = 64
    num_heads = 2
    num_layers = 1
    d_ff = 128
    max_seq_length = 40
    dropout = 0.1
    transformer_epochs = 5
    lr = 0.0001

    # Get static features
    cells, static_features = get_static_features("14850/data/link.csv")
    print("cells:", len(cells))
    # MAX_CELLS = 5000 # Run on subset of cells
    # cells = cells[:MAX_CELLS]
    # static_features = static_features[:MAX_CELLS]

    spatial_groups = build_spatial_groups(cells, neighbor_window=1)

    # Macronet hourly performance (speed, density, queue_ratio) or legacy CSV
    dynamic_features, T = get_dynamic_features("14850/data/td_link_performance.csv", cells)
    num_dynamic_features = dynamic_features.shape[-1]
    num_input_features = num_dynamic_features + static_features.shape[1]

    static_expanded = np.repeat(static_features[np.newaxis, :, :], T, axis=0)

    # Concatenate dynamic and static features
    X = np.concatenate([dynamic_features, static_expanded], axis=-1) # (T, MAX_CELLS, F)

    print("X:", X.shape)
    np.savetxt('14850/results/14850_features.csv', X[0, :, :], delimiter=',', fmt='%10.5f')

    # Normalize features
    X_mean = X.mean(axis=(0, 1), keepdims=True)
    X_std = X.std(axis=(0, 1), keepdims=True) + 1e-8
    X = (X - X_mean) / X_std    # (T, MAX_CELLS, F)

    # Get supervised training samples
    lookback_window = 3
    inputs, targets = get_training_samples(X, lookback_window, T)
    inputs = np.array(inputs)  # (B, lookback_window, N, F_in)
    targets_all = np.array(targets)# (B, N, F_out)
    targets = targets_all[:, :, :num_dynamic_features] # (B, N, F_out)
    print("targets:", targets.shape) # (10, 5000, 3)
    # np.savetxt('results/14850_targets_norm.csv', targets[0], delimiter=',', fmt='%10.5f')
    targets_real = targets_all * X_std[0, 0, :] + X_mean[0, 0, :]
    # np.savetxt('results/14850_targets.csv', targets_real[0,:,:num_dynamic_features], delimiter=',', fmt='%10.5f')

    dataset = TrafficDataset(inputs, targets)
    dataloader = DataLoader(dataset, batch_size=1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    criterion = nn.MSELoss()
    transformer = Transformer(
        num_input_features=num_input_features,
        num_predict_features=num_dynamic_features,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        d_ff=d_ff,
        p=dropout,
        spatial_groups=spatial_groups,
    ).to(device)
    optimizer = optim.Adam(transformer.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9)
    
    # Train transformer
    transformer_train_loss = train(transformer, dataloader, transformer_epochs, criterion, optimizer, device)

    # Predict next-step states
    sample_inputs, sample_labels = next(iter(dataloader))
    print("sample_inputs:", sample_inputs.shape) # (1, 3, 5000, 11)
    print("sample_labels:", sample_labels.shape) # (1, 5000, 3)
    mean_dyn = X_mean[:, :, :num_dynamic_features]
    std_dyn = X_std[:, :, :num_dynamic_features]

    target_norm = sample_labels[0].numpy()
    target_real = target_norm * std_dyn[0, 0, :] + mean_dyn[0, 0, :]
    print("target_real:", target_real.shape) # (5000, 3)
    np.savetxt('14850/results/14850_target_states.csv', target_real, delimiter=',', fmt='%10.5f')

    predicted_states = predict_next_states(transformer, sample_inputs, device)
    print("predicted_states:", predicted_states.shape) # (1, 5000, 3)

    pred_norm = predicted_states[0].numpy()
    pred_real = pred_norm * std_dyn[0, 0, :] + mean_dyn[0, 0, :]
    np.savetxt('14850/results/14850_predicted_states.csv', pred_real, delimiter=',', fmt='%10.5f')

    pred_loss = nn.MSELoss()(torch.tensor(target_norm), torch.tensor(pred_norm))
    print("Predicted loss:", pred_loss)
    print(pred_loss.item())



if __name__ == "__main__":
    main()