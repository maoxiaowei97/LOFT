import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PriorDecoderMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers=2):
        super().__init__()
        if n_layers < 1:
            raise ValueError("n_layers must be at least 1.")

        layers = []
        in_size = input_size
        for _ in range(n_layers - 1):
            layers.extend([nn.Linear(in_size, hidden_size), nn.ReLU(inplace=True)])
            in_size = hidden_size
        layers.append(nn.Linear(in_size, output_size))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def maybe_cat_exog(x, u, dim=-1):
    if u is None:
        return x
    if u.dim() == 3:
        u = u.unsqueeze(2)
    target_shape = list(x.shape)
    target_shape[dim] = u.shape[dim]
    u = u.expand(*target_shape)
    return torch.cat([x, u], dim=dim)


class LinearTemporalAttentionLayer(nn.Module):


    def __init__(self, d_model, kernel_dim, d_ff=None, dropout=0.0):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.w_q = nn.Linear(d_model, kernel_dim)
        self.w_k = nn.Linear(d_model, kernel_dim)
        self.value_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.MLP = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(),
                                 nn.Linear(d_ff, d_model))

    def forward(self, x):
        residual = x
        query = F.elu(self.w_q(x)) + 1.0
        key = F.elu(self.w_k(x)) + 1.0
        value = self.value_proj(x)

        core = torch.einsum('b n t m, b n t d -> b n m d', key, value)
        normalizer = torch.einsum('b n t m, b n m -> b n t',
                                  query, key.sum(dim=2)).clamp_min(1e-6)
        message = torch.einsum('b n t m, b n m d -> b n t d',
                               query, core) / normalizer.unsqueeze(-1)
        message = self.out_proj(message)

        message = residual + self.dropout(message)
        message = self.norm1(message)
        message = message + self.dropout(self.MLP(message))
        message = self.norm2(message)

        return message


class LinearSpatialAttentionLayer(nn.Module):


    def __init__(self, d_model, kernel_dim, d_ff=None, dropout=0.0):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.w_q = nn.Linear(d_model, kernel_dim)
        self.w_k = nn.Linear(d_model, kernel_dim)
        self.value_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.MLP = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(),
                                 nn.Linear(d_ff, d_model))

    def forward(self, x, emb=None, dim=1):
        x = x.transpose(dim, -2)
        residual = x
        query = F.elu(self.w_q(x)) + 1.0
        key = F.elu(self.w_k(x)) + 1.0
        value = self.value_proj(x)

        core = torch.einsum('... n m, ... n d -> ... m d', key, value)
        normalizer = torch.einsum('... n m, ... m -> ... n',
                                  query, key.sum(dim=-2)).clamp_min(1e-6)
        message = torch.einsum('... n m, ... m d -> ... n d',
                               query, core) / normalizer.unsqueeze(-1)
        message = self.out_proj(message)

        message = residual + self.dropout(message)
        message = self.norm1(message)
        message = message + self.dropout(self.MLP(message))
        message = self.norm2(message)

        return message.transpose(dim, -2)


class LowRankPriorEstimator(nn.Module):


    def __init__(
            self,
            num_nodes,
            input_dim=3,
            output_dim=1,
            input_embedding_dim=24,
            learnable_embedding_dim=80,
            feed_forward_dim=256,
            num_temporal_heads=4,
            num_layers=3,
            dropout=0.,
            windows=24,
            dim_proj=10,
            sigma_min=1e-3,
    ):
        super(LowRankPriorEstimator, self).__init__()

        self.num_nodes = num_nodes
        self.in_steps = windows
        self.out_steps = windows
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_embedding_dim = input_embedding_dim
        self.learnable_embedding_dim = learnable_embedding_dim
        self.model_dim = input_embedding_dim + learnable_embedding_dim
        self.num_temporal_heads = num_temporal_heads
        self.num_layers = num_layers
        self.prior_uncertainty_min = sigma_min

        self.input_proj = nn.Linear(input_dim, input_embedding_dim)
        self.dim_proj = dim_proj

        self.learnable_embedding = nn.init.xavier_uniform_(
            nn.Parameter(torch.empty(windows, num_nodes, learnable_embedding_dim)))

        self.prior_mean_decoder = PriorDecoderMLP(self.model_dim, self.model_dim, output_dim, n_layers=2)
        self.prior_uncertainty_decoder = PriorDecoderMLP(self.model_dim, self.model_dim, output_dim, n_layers=2)

        self.attn_layers_t = nn.ModuleList(
            [LinearTemporalAttentionLayer(self.model_dim, self.dim_proj, self.model_dim, dropout)
             for _ in range(num_layers)])

        self.attn_layers_s = nn.ModuleList(
            [LinearSpatialAttentionLayer(self.model_dim, self.dim_proj, self.model_dim, dropout)
             for _ in range(num_layers)])

    def forward(self, x, u, mask):
        batch_size = x.shape[0]
        x = x * mask

        x = maybe_cat_exog(x, u)
        x = self.input_proj(x)

        node_emb = self.learnable_embedding.expand(batch_size, *self.learnable_embedding.shape)
        x = torch.cat([x, node_emb], dim=-1)

        x = x.permute(0, 2, 1, 3)
        for att_t, att_s in zip(self.attn_layers_t, self.attn_layers_s):
            x = att_t(x)
            x = att_s(x, self.learnable_embedding, dim=1)

        x = x.permute(0, 2, 1, 3)
        prior_mean = self.prior_mean_decoder(x)
        prior_uncertainty = F.softplus(self.prior_uncertainty_decoder(x)) + self.prior_uncertainty_min

        return prior_mean, prior_uncertainty
