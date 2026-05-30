from functools import lru_cache

import pywt
import torch
import torch.nn.functional as F
from torch import nn


@lru_cache(maxsize=16)
def _get_real_wavelet_filters(wavelet):
    w = pywt.Wavelet(wavelet)
    dec_lo = torch.tensor(w.dec_lo[::-1], dtype=torch.float32)
    dec_hi = torch.tensor(w.dec_hi[::-1], dtype=torch.float32)
    rec_lo = torch.tensor(w.rec_lo, dtype=torch.float32)
    rec_hi = torch.tensor(w.rec_hi, dtype=torch.float32)
    return dec_lo, dec_hi, rec_lo, rec_hi


@lru_cache(maxsize=16)
def _get_real_wavelet_banks(wavelet):
    dec_lo, dec_hi, rec_lo, rec_hi = _get_real_wavelet_filters(wavelet)
    analysis_bank = torch.stack([
        torch.outer(dec_hi, dec_hi),
        torch.outer(dec_hi, dec_lo),
        torch.outer(dec_lo, dec_hi),
        torch.outer(dec_lo, dec_lo),
    ], dim=0).unsqueeze(1)
    synthesis_bank = torch.stack([
        torch.outer(rec_hi, rec_hi),
        torch.outer(rec_hi, rec_lo),
        torch.outer(rec_lo, rec_hi),
        torch.outer(rec_lo, rec_lo),
    ], dim=0).unsqueeze(1)
    pad = max(analysis_bank.shape[-1] // 2 - 1, 0)
    return analysis_bank, synthesis_bank, pad


class RealWaveletTransform2D(nn.Module):
    def __init__(self, channels, wavelet="db2", padding_mode="reflect"):
        super().__init__()
        self.wavelet = wavelet
        self.padding_mode = padding_mode
        self.channels = channels

        analysis_bank, synthesis_bank, pad = _get_real_wavelet_banks(self.wavelet)
        self.register_buffer(
            "analysis_bank",
            analysis_bank.repeat(channels, 1, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "synthesis_bank",
            synthesis_bank.repeat(channels, 1, 1, 1),
            persistent=False,
        )
        self.pad = pad

    def dwt2d(self, x, split_subbands=True):
        if x.shape[1] != self.channels:
            raise ValueError(
                f"Expected {self.channels} channels, but got {x.shape[1]} channels."
            )

        if self.pad > 0:
            x = F.pad(x, (self.pad, self.pad, self.pad, self.pad), mode=self.padding_mode)

        analysis_bank = self.analysis_bank
        if analysis_bank.dtype != x.dtype:
            analysis_bank = analysis_bank.to(dtype=x.dtype)

        coeffs = F.conv2d(x, analysis_bank, stride=2, groups=self.channels)
        if split_subbands:
            return torch.chunk(coeffs, 4, dim=1)
        return coeffs

    def idwt2d(self, hh, hl=None, lh=None, ll=None):
        if hl is None:
            coeffs = hh
        else:
            coeffs = torch.cat([hh, hl, lh, ll], dim=1)

        expected_channels = self.channels * 4
        if coeffs.shape[1] != expected_channels:
            raise ValueError(
                f"Expected {expected_channels} packed subband channels, but got {coeffs.shape[1]}."
            )

        synthesis_bank = self.synthesis_bank
        if synthesis_bank.dtype != coeffs.dtype:
            synthesis_bank = synthesis_bank.to(dtype=coeffs.dtype)

        recon = F.conv_transpose2d(coeffs, synthesis_bank, stride=2, groups=self.channels)
        if self.pad > 0:
            recon = recon[..., self.pad:-self.pad, self.pad:-self.pad]
        return recon


class SpatialMHSA2D(nn.Module):
    def __init__(
        self,
        channels,
        num_heads=4,
        mlp_ratio=2.0,
        dropout=0.0,
        attention_axis="row",
        patch_size=8,
        rope_base=10000.0,
    ):
        super().__init__()
        if channels % num_heads != 0:
            raise ValueError(
                f"Expected channels ({channels}) to be divisible by num_heads ({num_heads})."
            )
        if attention_axis not in {"row", "col"}:
            raise ValueError(
                f"Unsupported attention_axis: {attention_axis}. Expected 'row' or 'col'."
            )
        if patch_size <= 0:
            raise ValueError(f"Expected patch_size > 0, but got {patch_size}.")

        hidden_dim = int(channels * mlp_ratio)
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        if self.head_dim % 2 != 0:
            raise ValueError(
                f"Expected per-head dimension ({self.head_dim}) to be even for RoPE."
            )
        if self.head_dim < 4:
            raise ValueError(
                f"Expected per-head dimension ({self.head_dim}) >= 4 to encode both row and column RoPE."
            )
        self.dropout = dropout
        self.attention_axis = attention_axis
        self.patch_size = patch_size
        self.rope_base = rope_base
        self.rope_cache_key = None

        self.register_buffer("row_rope_cos", torch.empty(0), persistent=False)
        self.register_buffer("row_rope_sin", torch.empty(0), persistent=False)
        self.register_buffer("col_rope_cos", torch.empty(0), persistent=False)
        self.register_buffer("col_rope_sin", torch.empty(0), persistent=False)

        self.norm1 = nn.LayerNorm(channels)
        self.qkv = nn.Linear(channels, channels * 3)
        self.proj = nn.Linear(channels, channels)
        self.proj_dropout = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(channels)
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, channels),
            nn.Dropout(dropout),
        )

    def _build_rope_cache(self, height, width, device, dtype):
        cache_key = (height, width, device, dtype)
        if self.rope_cache_key == cache_key:
            return

        pair_count = self.head_dim // 2
        inv_freq = self.rope_base ** (
            -torch.arange(pair_count, device=device, dtype=torch.float32) / max(pair_count, 1)
        )
        row_positions = torch.arange(height, device=device, dtype=torch.float32)
        col_positions = torch.arange(width, device=device, dtype=torch.float32)

        row_angles = row_positions[:, None] * inv_freq[None, :]
        col_angles = col_positions[:, None] * inv_freq[None, :]
        row_angles = row_angles.repeat_interleave(2, dim=-1)
        col_angles = col_angles.repeat_interleave(2, dim=-1)

        self.row_rope_cos = row_angles.cos().to(dtype=dtype)
        self.row_rope_sin = row_angles.sin().to(dtype=dtype)
        self.col_rope_cos = col_angles.cos().to(dtype=dtype)
        self.col_rope_sin = col_angles.sin().to(dtype=dtype)
        self.rope_cache_key = cache_key

    def _reshape_to_axis_patches(self, tokens, height, width):
        batch_size = tokens.shape[0]

        if self.attention_axis == "row":
            if width % self.patch_size != 0:
                raise ValueError(
                    f"Expected width ({width}) to be divisible by patch_size ({self.patch_size}) "
                    "for row-wise attention."
                )
            num_patches = width // self.patch_size
            tokens = tokens.view(batch_size, height, width, self.channels)
            tokens = tokens.view(batch_size, height, num_patches, self.patch_size, self.channels)
        else:
            if height % self.patch_size != 0:
                raise ValueError(
                    f"Expected height ({height}) to be divisible by patch_size ({self.patch_size}) "
                    "for column-wise attention."
                )
            num_patches = height // self.patch_size
            tokens = tokens.view(batch_size, height, width, self.channels)
            tokens = tokens.view(batch_size, num_patches, self.patch_size, width, self.channels)
            tokens = tokens.permute(0, 3, 1, 2, 4)

        num_sequences = (
            batch_size * height * num_patches
            if self.attention_axis == "row"
            else batch_size * width * num_patches
        )
        return tokens.reshape(num_sequences, self.patch_size, self.channels)

    def _reshape_rope_to_axis_patches(self, batch_size, height, width):
        if self.attention_axis == "row":
            num_patches = width // self.patch_size
            cos = self.col_rope_cos.view(num_patches, self.patch_size, self.head_dim)
            sin = self.col_rope_sin.view(num_patches, self.patch_size, self.head_dim)
            cos = cos.unsqueeze(0).unsqueeze(0).expand(
                batch_size, height, num_patches, self.patch_size, self.head_dim
            )
            sin = sin.unsqueeze(0).unsqueeze(0).expand(
                batch_size, height, num_patches, self.patch_size, self.head_dim
            )
        else:
            num_patches = height // self.patch_size
            cos = self.row_rope_cos.view(num_patches, self.patch_size, self.head_dim)
            sin = self.row_rope_sin.view(num_patches, self.patch_size, self.head_dim)
            cos = cos.unsqueeze(0).unsqueeze(0).expand(
                batch_size, width, num_patches, self.patch_size, self.head_dim
            )
            sin = sin.unsqueeze(0).unsqueeze(0).expand(
                batch_size, width, num_patches, self.patch_size, self.head_dim
            )

        return (
            cos.reshape(-1, self.patch_size, self.head_dim),
            sin.reshape(-1, self.patch_size, self.head_dim),
        )

    @staticmethod
    def _rotate_half(x):
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        return torch.stack((-x_odd, x_even), dim=-1).flatten(-2)

    def _apply_rope(self, q, k, cos, sin):
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
        q = (q * cos) + (self._rotate_half(q) * sin)
        k = (k * cos) + (self._rotate_half(k) * sin)
        return q, k

    def _reshape_from_axis_patches(self, tokens, batch_size, height, width):
        if self.attention_axis == "row":
            num_patches = width // self.patch_size
            tokens = tokens.view(batch_size, height, num_patches, self.patch_size, self.channels)
            tokens = tokens.reshape(batch_size, height, width, self.channels)
        else:
            num_patches = height // self.patch_size
            tokens = tokens.view(batch_size, width, num_patches, self.patch_size, self.channels)
            tokens = tokens.permute(0, 2, 3, 1, 4).reshape(batch_size, height, width, self.channels)

        return tokens.reshape(batch_size, height * width, self.channels)

    def forward(self, x):
        batch_size, channels, height, width = x.shape
        tokens = x.flatten(2).transpose(1, 2)

        attn_input = self.norm1(tokens)
        attn_input = self._reshape_to_axis_patches(attn_input, height, width)
        qkv = self.qkv(attn_input)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(-1, self.patch_size, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(-1, self.patch_size, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(-1, self.patch_size, self.num_heads, self.head_dim).transpose(1, 2)

        self._build_rope_cache(height, width, x.device, q.dtype)
        rope_cos, rope_sin = self._reshape_rope_to_axis_patches(batch_size, height, width)
        q, k = self._apply_rope(q, k, rope_cos, rope_sin)

        attn_output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.dropout if self.training else 0.0,
        )
        attn_output = attn_output.transpose(1, 2).contiguous().view(-1, self.patch_size, channels)
        attn_output = self.proj_dropout(self.proj(attn_output))
        attn_output = self._reshape_from_axis_patches(attn_output, batch_size, height, width)

        tokens = tokens + attn_output
        tokens = tokens + self.mlp(self.norm2(tokens))

        return tokens.transpose(1, 2).reshape(batch_size, channels, height, width)


class ResidualBlock2D(nn.Module):
    def __init__(
        self,
        in_channel=1,
        out_channel=1,
        wavelet="db2",
        subband_fuse_type="conv1x1",
        attention_axis="row",
        attention_patch_size=8,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.subband_fuse_type = subband_fuse_type
        subband_channels = in_channel * 4
        num_heads = min(4, subband_channels)
        while subband_channels % num_heads != 0 and num_heads > 1:
            num_heads -= 1

        self.wavelet_enhance = RealWaveletTransform2D(channels=in_channel, wavelet=wavelet)
        if subband_fuse_type == "mhsa":
            self.subband_fuse_row = SpatialMHSA2D(
                channels=subband_channels,
                num_heads=num_heads,
                attention_axis="row",
                patch_size=attention_patch_size,
            )
            self.subband_fuse_col = SpatialMHSA2D(
                channels=subband_channels,
                num_heads=num_heads,
                attention_axis="col",
                patch_size=attention_patch_size,
            )
        elif subband_fuse_type == "conv1x1":
            self.subband_fuse = nn.Conv2d(subband_channels, subband_channels, kernel_size=1)
        else:
            raise ValueError(
                f"Unsupported subband_fuse_type: {subband_fuse_type}. Expected 'mhsa' or 'conv1x1'."
            )
        self.shortcut_adjust = (
            nn.Identity()
            if in_channel == out_channel
            else nn.Conv2d(in_channel, out_channel, kernel_size=1)
        )

        self.shortcut = (
            nn.Identity()
            if in_channel == out_channel
            else nn.Conv2d(in_channel, out_channel, kernel_size=1)
        )

    def forward(self, x):
        if self.subband_fuse_type == "mhsa":
            fused_subbands = self.subband_fuse_row(self.wavelet_enhance.dwt2d(x, split_subbands=False))
            fused_subbands = self.subband_fuse_col(fused_subbands)
        elif self.subband_fuse_type == "conv1x1":
            fused_subbands = self.subband_fuse(self.wavelet_enhance.dwt2d(x, split_subbands=False))
        identity = self.wavelet_enhance.idwt2d(fused_subbands)
        identity = self.shortcut_adjust(identity)
        identity_x = self.shortcut(x)
        residual = self.bn2(self.conv2(self.prelu(self.bn1(self.conv1(x)))))
        return identity + residual + identity_x

# class ResidualBlock2D(nn.Module):
#     def __init__(self, in_channel=1, out_channel=1):
#         super().__init__()
#         self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1)
#         self.bn1 = nn.BatchNorm2d(out_channel)
#         self.prelu = nn.PReLU()
#         self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1)
#         self.bn2 = nn.BatchNorm2d(out_channel)
#         self.shortcut = (
#             nn.Identity()
#             if in_channel == out_channel
#             else nn.Conv2d(in_channel, out_channel, kernel_size=1)
#         )
#
#     def forward(self, x):
#         identity = self.shortcut(x)
#         residual = self.conv1(x)
#         residual = self.bn1(residual)
#         residual = self.prelu(residual)
#         residual = self.conv2(residual)
#         residual = self.bn2(residual)
#         return identity + residual


class radar_2d_wavelet(nn.Module):
    def __init__(self, scale_factor, input_dim=1):
        super().__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(8, 8, kernel_size=9, padding=4),
            nn.PReLU()
        )
        self.block2 = ResidualBlock2D(8, 32, subband_fuse_type="mhsa")
        self.block3 = ResidualBlock2D(32, 64)
        self.block4 = ResidualBlock2D(64, 128, subband_fuse_type="mhsa")
        self.block5 = ResidualBlock2D(128, 64)
        self.block6 = ResidualBlock2D(64, 32, subband_fuse_type="mhsa")
        self.block7 = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16)
        )
        self.block1_skip = nn.Conv2d(8, 16, kernel_size=1)

        self.input_dim = input_dim
        self.scale_factor = scale_factor
        self.block8 = nn.Sequential(
            nn.Conv2d(16, 24, kernel_size=9, padding=4)
        )

    def forward(self, x):
        batch_size, complex_dim, channels, height, width = x.shape
        x = x.flatten(1, 2)

        block1 = self.block1(x)
        block2 = self.block2(block1)
        block3 = self.block3(block2)
        block4 = self.block4(block3)
        block5 = self.block5(block4)
        block6 = self.block6(block5)
        block7 = self.block7(block6)
        block1_skip = self.block1_skip(block1)
        block8 = self.block8(block1_skip + block7)

        block8 = block8.reshape(
            batch_size,
            complex_dim,
            int(channels * self.scale_factor),
            height,
            width
        )

        return block8
