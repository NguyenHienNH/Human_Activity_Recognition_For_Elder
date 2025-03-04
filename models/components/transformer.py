import torch
import torch.nn as nn

class SwinTransformerLayer(nn.Module):
    def __init__(self, dim, num_heads, window_size=None, shift_size=0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads

        # Tự động điều chỉnh window_size nếu không được cung cấp
        self.window_size = window_size if window_size is not None else 8
        self.shift_size = shift_size

        self.attn = nn.MultiheadAttention(dim, num_heads)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.mlp = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Linear(4 * dim, dim)
        )

    def forward(self, x):
        # Đảm bảo x có định dạng đúng
        original_shape = x.shape

        # Xử lý tensor đầu vào đúng cách
        if len(x.shape) == 2:
            B, C = x.shape
            x = x.unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)
        elif len(x.shape) == 3:
            B, C, L = x.shape
            x = x.unsqueeze(-1)  # (B, C, L, 1)

        B, C, T, J = x.shape

        # Áp dụng self-attention
        residual = x

        # Flatten và permute để phù hợp với multi-head attention
        x_flat = x.permute(0, 2, 3, 1).reshape(B, T*J, C)  # (B, T*J, C)
        x_flat = self.norm1(x_flat)

        # Áp dụng multi-head attention
        attn_out, _ = self.attn(x_flat, x_flat, x_flat)

        # Reshape lại và thêm residual connection
        attn_out = attn_out.reshape(B, T, J, C).permute(0, 3, 1, 2)  # (B, C, T, J)
        x = residual + attn_out

        # Feed-forward network
        x_ffn = x.permute(0, 2, 3, 1).reshape(B, T*J, C)  # (B, T*J, C)
        x_ffn = self.norm2(x_ffn)
        x_ffn = self.mlp(x_ffn)
        x_ffn = x_ffn.reshape(B, T, J, C).permute(0, 3, 1, 2)  # (B, C, T, J)

        x = x + x_ffn

        # Trả về tensor có cùng hình dạng với đầu vào
        if len(original_shape) < 4:
            if len(original_shape) == 3:
                x = x.squeeze(-1)  # (B, C, T)
            elif len(original_shape) == 2:
                x = x.squeeze(-1).squeeze(-1)  # (B, C)

        return x

class StochasticDepth(nn.Module):
    def __init__(self, drop_prob=0.1):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if not self.training or self.drop_prob == 0:
            return x

        keep_prob = 1 - self.drop_prob
        mask = x.new_empty([x.shape[0], 1, 1, 1]).bernoulli_(keep_prob)
        return x / keep_prob * mask