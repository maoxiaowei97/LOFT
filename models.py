import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def get_torch_trans(heads=8, layers=1, channels=64):
    encoder_layer = nn.TransformerEncoderLayer(
        d_model=channels, nhead=heads, dim_feedforward=64, activation="gelu"
    )
    return nn.TransformerEncoder(encoder_layer, num_layers=layers)


def Conv1d_with_init(in_channels, out_channels, kernel_size):
    layer = nn.Conv1d(in_channels, out_channels, kernel_size)
    nn.init.kaiming_normal_(layer.weight)
    return layer


class IntgTimeEmbedding(nn.Module):
    def __init__(self, num_steps, embedding_dim=128, projection_dim=None):
        super().__init__()
        if projection_dim is None:
            projection_dim = embedding_dim
        self.register_buffer(
            "embedding",
            self._build_embedding(num_steps, embedding_dim / 2),
            persistent=False,
        )
        self.projection1 = nn.Linear(embedding_dim, projection_dim)
        self.projection2 = nn.Linear(projection_dim, projection_dim)

    def forward(self, intg_time_step):
        x = self.embedding[intg_time_step]
        x = self.projection1(x)
        x = F.silu(x)
        x = self.projection2(x)
        x = F.silu(x)
        return x

    def _build_embedding(self, num_steps, dim=64):
        steps = torch.arange(num_steps).unsqueeze(1)
        frequencies = 10.0 ** (torch.arange(dim) / (dim - 1) * 4.0).unsqueeze(0)
        table = steps * frequencies
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)
        return table

class LOFTNet(nn.Module):
    def __init__(self, config, inputdim=2):
        super().__init__()
        self.channels = int(config["channels"])

        self.intg_time_embedding = IntgTimeEmbedding(
            num_steps=int(config["num_steps"]),
            embedding_dim=int(config["intg_time_embedding_dim"]),
        )

        self.input_projection = Conv1d_with_init(inputdim, self.channels, 1)
        self.output_projection1 = Conv1d_with_init(self.channels, self.channels, 1)

        self.output_projection2 = Conv1d_with_init(self.channels, 1, 1)
        nn.init.zeros_(self.output_projection2.weight)
        
        self.residual_layers = nn.ModuleList(
            [
                ResidualBlock(
                    side_dim=int(config["side_dim"]),
                    channels=self.channels,
                    intg_time_embedding_dim=int(config["intg_time_embedding_dim"]),
                    nheads=int(config["nheads"]),
                )
                for _ in range(int(config["layers"]))
            ]
        )

    def forward(self, x, cond_info, intg_time_step):
        B, inputdim, K, L = x.shape
        x = x.reshape(B, inputdim, K * L)
        x = self.input_projection(x)
        x = F.relu(x)
        x = x.reshape(B, self.channels, K, L)
        
        intg_time_emb = self.intg_time_embedding(intg_time_step)
        
        skip = []
        attn_weights = None
        for layer in self.residual_layers:

            x, skip_connection, attn_weights = layer(x, cond_info, intg_time_emb)
            skip.append(skip_connection)

        x = torch.sum(torch.stack(skip), dim=0) / math.sqrt(len(self.residual_layers))
        x = x.reshape(B, self.channels, K * L)
        x = self.output_projection1(x)
        x = F.relu(x)
        x = self.output_projection2(x)
        x = x.reshape(B, K, L)

        return x, attn_weights


class FlashAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        B, L, C = x.shape
        qkv = self.qkv(x).reshape(B, L, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        out = F.scaled_dot_product_attention(q, k, v)
        
        out = out.transpose(1, 2).reshape(B, L, C)
        out = self.out_proj(out)
        return out, None


class ResidualBlock(nn.Module):
    def __init__(self, side_dim, channels, intg_time_embedding_dim, nheads):
        super().__init__()
        self.intg_time_projection = nn.Linear(intg_time_embedding_dim, channels)
        self.cond_projection = Conv1d_with_init(side_dim, 2 * channels, 1)
        self.mid_projection = Conv1d_with_init(channels, 2 * channels, 1)
        self.output_projection = Conv1d_with_init(channels, 2 * channels, 1)

        self.feature_attn = FlashAttention(channels, nheads)
        self.feature_ffn = nn.Sequential(
            nn.Linear(channels, channels * 4),
            nn.GELU(),
            nn.Linear(channels * 4, channels),
        )
        self.feature_norm1 = nn.LayerNorm(channels)
        self.feature_norm2 = nn.LayerNorm(channels)
        
        self.time_layer = get_torch_trans(heads=nheads, layers=1, channels=channels)


    def forward_time(self, y, base_shape):
        B, channel, K, L = base_shape
        if L == 1:
            return y
        y = y.reshape(B, channel, K, L).permute(0, 2, 1, 3).reshape(B * K, channel, L)
        y = self.time_layer(y.permute(2, 0, 1)).permute(1, 2, 0)
        y = y.reshape(B, K, channel, L).permute(0, 2, 1, 3).reshape(B, channel, K * L)
        return y

    def forward_feature(self, y, base_shape):
        B, channel, K, L = base_shape
        if K == 1:
            return y, None
            
        y = y.reshape(B, channel, K, L).permute(0, 3, 2, 1).reshape(B * L, K, channel)

        y_norm = self.feature_norm1(y)
        attn_output, attn_weights = self.feature_attn(y_norm)
        y = y + attn_output
        
        ffn_output = self.feature_ffn(self.feature_norm2(y))
        y = y + ffn_output

        y = y.reshape(B, L, K, channel).permute(0, 3, 2, 1).reshape(B, channel, K * L)

        if attn_weights is not None:
            attn_weights = attn_weights.view(B, L, K, K).mean(dim=1)
        
        return y, attn_weights

    def forward(self, x, cond_info, intg_time_emb):
        B, channel, K, L = x.shape
        base_shape = x.shape
        x = x.reshape(B, channel, K * L)

        intg_time_emb = self.intg_time_projection(intg_time_emb).unsqueeze(-1)
        y = x + intg_time_emb

        y = self.forward_time(y, base_shape)
        y, attn_weights = self.forward_feature(y, base_shape)
        y = self.mid_projection(y)

        _, cond_dim, _, _ = cond_info.shape
        cond_info = cond_info.reshape(B, cond_dim, K * L)
        cond_info = self.cond_projection(cond_info)
        y = y + cond_info
        
        gate, filter = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filter)
        y = self.output_projection(y)

        residual, skip = torch.chunk(y, 2, dim=1)
        x = x.reshape(base_shape)
        residual = residual.reshape(base_shape)
        skip = skip.reshape(base_shape)

        return (x + residual) / math.sqrt(2.0), skip, attn_weights
