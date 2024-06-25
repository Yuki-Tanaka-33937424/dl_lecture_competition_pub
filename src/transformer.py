import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange
from torch import einsum

class ConvBlock1D(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.proj = nn.Conv1d(in_dim, out_dim, kernel_size=3, stride=1, padding=1)
        self.norm = nn.BatchNorm1d(out_dim)
        self.act = nn.SiLU()
    
    def forward(self, x):
        x = self.proj(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class SinusoidalPositionEmbeddings1D(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
 
    def forward(self, time):
        device = time.device 
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class Attention1D(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** (-0.5)
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv1d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv1d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, l = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, "b (h c) l -> b h c l", h=self.heads), qkv)
        q = q * self.scale
        sim = einsum("b h d i, b h d j -> b h i j", q, k)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)
        out = einsum("b h i j, b h d j -> b h i d", attn, v)
        out = rearrange(out, "b h l d -> b (h d) l", l=l)
        return self.to_out(out)


class FeedForward(nn.Module):
    def __init__(self, dim, d_ff=120, dropout=0.1):
        super().__init__()
        self.linear_1 = nn.Linear(dim, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, dim)

    def forward(self, x):
        x = self.linear_1(x)
        x = self.dropout(F.relu(x))
        x = self.linear_2(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, dim, length, dropout=0.1):
        super().__init__()
        self.pos_emb = SinusoidalPositionEmbeddings1D(dim=dim)
        self.norm_1 = nn.LayerNorm((dim, length))
        self.norm_2 = nn.LayerNorm((dim, length))
        self.attn = Attention1D(dim)
        self.ff = FeedForward(dim)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x):
        time = torch.arange(x.size()[-1], device=x.device)
        time_emb = self.pos_emb(time=time)
        time_emb = time_emb.transpose(0, 1).to(x.device)
        x1 = x + time_emb.unsqueeze(0)
        x_normlized = self.norm_1(x1)
        output = self.attn(x_normlized)
        x2 = x1 + self.dropout_1(output)
        x_normalized2 = self.norm_2(x2)
        x_normalized2 = rearrange(x_normalized2, "b c l -> b l c")
        x_normalized2 = self.dropout_2(self.ff(x_normalized2))
        x_normalized2 = rearrange(x_normalized2, "b l c -> b c l")
        output = x2 + x_normalized2
        return output


class BasicTransformerClassifier(nn.Module):
    def __init__(self, num_classes: int, seq_len: int, in_channels: int, dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Conv1d(in_channels, dim, kernel_size=1)
        self.conv1 = ConvBlock1D(dim, dim * 2)
        self.conv2 = ConvBlock1D(dim * 2, dim * 4)
        self.transformer1 = TransformerBlock(dim=dim * 2, length=seq_len, dropout=dropout)
        self.transformer2 = TransformerBlock(dim=dim * 4, length=seq_len, dropout=dropout)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            Rearrange("b d 1 -> b d"),
            nn.Linear(dim * 4, num_classes),
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Args:
            X (b, c, t): Input tensor with batch size b, channels c, and sequence length t
        Returns:
            X (b, num_classes): Output tensor with class probabilities
        """
        X = self.proj(X)
        X = self.conv1(X)
        X = self.transformer1(X)
        X = self.conv2(X)
        X = self.transformer2(X)
        return self.head(X)