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


class SubbandMultiHeadAttention2D(nn.Module):
    def __init__(self, channels, num_heads=4):
        super().__init__()
        embed_dim = channels * 4
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"Expected channels*4 ({embed_dim}) to be divisible by num_heads ({num_heads})."
            )

        self.norm = nn.LayerNorm(embed_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.PReLU(),
        )

    def forward(self, x):
        batch_size, channels, height, width = x.shape
        tokens = x.flatten(2).transpose(1, 2)
        norm_tokens = self.norm(tokens)
        attended_tokens, _ = self.attention(norm_tokens, norm_tokens, norm_tokens)
        fused_tokens = tokens + self.proj(attended_tokens)
        return fused_tokens.transpose(1, 2).reshape(batch_size, channels, height, width)


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


class ResidualBlock2D(nn.Module):
    def __init__(self, in_channel=1, out_channel=1, wavelet="db2"):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channel)

        self.wavelet_enhance = RealWaveletTransform2D(channels=in_channel, wavelet=wavelet)
        self.subband_fuse = SubbandMultiHeadAttention2D(channels=in_channel, num_heads=4)
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
        fused_subbands = self.subband_fuse(self.wavelet_enhance.dwt2d(x, split_subbands=False))
        identity = self.wavelet_enhance.idwt2d(fused_subbands)
        identity = self.shortcut_adjust(identity)
        identity_x = self.shortcut(x)
        residual = self.bn2(self.conv2(self.prelu(self.bn1(self.conv1(x)))))
        return identity + residual + identity_x


class Generator_radar2D_adc(nn.Module):
    def __init__(self, scale_factor, input_dim=1):
        super().__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(8, 8, kernel_size=9, padding=4),
            nn.PReLU()
        )
        self.block2 = ResidualBlock2D(8, 32)
        self.block3 = ResidualBlock2D(32, 64)
        self.block4 = ResidualBlock2D(64, 128)
        self.block5 = ResidualBlock2D(128, 64)
        self.block6 = ResidualBlock2D(64, 32)
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
