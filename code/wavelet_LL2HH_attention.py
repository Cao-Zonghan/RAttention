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
        self.dropout = dropout
        self.attention_axis = attention_axis
        self.patch_size = patch_size

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

        return tokens.reshape(batch_size * height * num_patches if self.attention_axis == "row" else batch_size * width * num_patches, self.patch_size, self.channels)

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


class RMSNorm(nn.Module):
    """Channel-last RMSNorm used on token tensors: [B, N, C]."""

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # Keep variance computation in fp32 for stable AMP/fp16 training.
        variance = x.float().pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance.to(dtype=x.dtype) + self.eps)
        return x * self.weight


class DepthwiseSeparableSubbandEncoder(nn.Module):
    """Lightweight per-subband directional encoder.

    Shape: [B, C, H, W] -> [B, C, H, W]
    Depthwise 3x3 preserves locality and pointwise 1x1 mixes channels inside one
    subband only, so HH/HL/LH/LL semantics are not mixed prematurely.
    """

    def __init__(self, channels, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
        )

    def forward(self, x):
        return x + self.net(x)


class TokenLearner2D(nn.Module):
    """TokenLearner-style soft spatial aggregation.

    Produces a fixed small set of latent tokens from a dynamic HxW feature map
    without sorting/top-k. The attention maps are computed in fp32 softmax for
    numerical stability, then cast back for efficient matmul.
    """

    def __init__(self, channels, num_tokens=16):
        super().__init__()
        if num_tokens <= 0:
            raise ValueError(f"Expected num_tokens > 0, but got {num_tokens}.")
        self.num_tokens = num_tokens
        hidden = max(8, min(channels, 64))
        self.selector = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=True),
            nn.GELU(),
            nn.Conv2d(hidden, num_tokens, kernel_size=1, bias=True),
        )

    def forward(self, x):
        # x: [B, C, H, W]
        batch_size, channels, height, width = x.shape
        spatial_tokens = height * width

        # weights: [B, L, HW], features: [B, HW, C]
        weights = self.selector(x).flatten(2)
        weights = weights.float().softmax(dim=-1).to(dtype=x.dtype)
        features = x.flatten(2).transpose(1, 2)

        # latents: [B, L, C]
        latents = torch.bmm(weights, features)
        return latents, weights, (height, width, spatial_tokens)


class FrequencyGate2D(nn.Module):
    """Extremely lightweight frequency-aware SE gate over HH/HL/LH/LL."""

    def __init__(self, subband_channels):
        super().__init__()
        self.score = nn.Conv2d(subband_channels, 1, kernel_size=1, bias=True)

    def forward(self, hh, hl, lh, ll):
        # Per-subband descriptor: [B, 4]
        descriptors = torch.cat(
            [
                self.score(hh).mean(dim=(2, 3)),
                self.score(hl).mean(dim=(2, 3)),
                self.score(lh).mean(dim=(2, 3)),
                self.score(ll).mean(dim=(2, 3)),
            ],
            dim=1,
        )
        gates = descriptors.float().softmax(dim=1).to(dtype=hh.dtype).view(-1, 4, 1, 1, 1)
        return hh * gates[:, 0], hl * gates[:, 1], lh * gates[:, 2], ll * gates[:, 3]


class WaveletSubbandMHSA2D(nn.Module):
    """Efficient wavelet-aware latent cross-attention block.

    Design summary:
      1. Keep HH/HL/LH/LL as separate semantic streams.
      2. Encode each subband with depthwise-separable local filtering.
      3. Compress each subband to a small TokenLearner latent set.
      4. Use high-frequency latent queries (HH/HL/LH) attending to LL latent memory.
      5. Scatter latent updates back to dense maps with the same soft assignment.
      6. Finish with lightweight large-kernel depthwise spatial mixing and FFN.

    Input/Output shape: [B, 4*C, H, W]
    The class keeps the original constructor arguments so it drops into the
    existing ResidualBlock2D without call-site changes. attention_axis is used
    to select anisotropic spatial mixing (row-like or column-like).
    """

    def __init__(
        self,
        channels,
        num_heads=4,
        mlp_ratio=2.0,
        dropout=0.0,
        attention_axis="row",
        patch_size=8,
        latent_tokens=None,
    ):
        super().__init__()
        if channels % 4 != 0:
            raise ValueError(
                f"Expected packed wavelet channels ({channels}) to be divisible by 4."
            )
        if attention_axis not in {"row", "col"}:
            raise ValueError(
                f"Unsupported attention_axis: {attention_axis}. Expected 'row' or 'col'."
            )
        if patch_size <= 0:
            raise ValueError(f"Expected patch_size > 0, but got {patch_size}.")

        self.channels = channels
        self.subband_channels = channels // 4
        self.num_heads = min(num_heads, self.subband_channels)
        while self.subband_channels % self.num_heads != 0 and self.num_heads > 1:
            self.num_heads -= 1
        self.head_dim = self.subband_channels // self.num_heads
        self.dropout = dropout
        self.attention_axis = attention_axis
        self.patch_size = patch_size
        self.latent_tokens = latent_tokens or max(8, min(32, patch_size * 2))
        self.residual_scale = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))

        # Per-frequency local encoders. They preserve directionality before any
        # global interaction and are cheap because the 3x3 stage is depthwise.
        self.hh_encoder = DepthwiseSeparableSubbandEncoder(self.subband_channels, dropout)
        self.hl_encoder = DepthwiseSeparableSubbandEncoder(self.subband_channels, dropout)
        self.lh_encoder = DepthwiseSeparableSubbandEncoder(self.subband_channels, dropout)
        self.ll_encoder = DepthwiseSeparableSubbandEncoder(self.subband_channels, dropout)

        # Token compression: HW -> L latents per subband, where L is usually 8-32.
        self.hh_tokens = TokenLearner2D(self.subband_channels, self.latent_tokens)
        self.hl_tokens = TokenLearner2D(self.subband_channels, self.latent_tokens)
        self.lh_tokens = TokenLearner2D(self.subband_channels, self.latent_tokens)
        self.ll_tokens = TokenLearner2D(self.subband_channels, self.latent_tokens)

        self.freq_gate = FrequencyGate2D(self.subband_channels)

        # Shared projections keep parameters small and align the three high bands
        # to a common LL semantic memory space.
        self.q_norm = RMSNorm(self.subband_channels)
        self.kv_norm = RMSNorm(self.subband_channels)
        self.q_proj = nn.Linear(self.subband_channels, self.subband_channels, bias=False)
        self.kv_proj = nn.Linear(self.subband_channels, self.subband_channels * 2, bias=False)
        self.out_proj = nn.Linear(self.subband_channels, self.subband_channels, bias=False)
        self.attn_dropout = dropout
        self.proj_dropout = nn.Dropout(dropout)

        # Axis-aware long-range spatial mixing in dense space. This preserves the
        # old row/column inductive bias without quadratic axial attention.
        if attention_axis == "row":
            kernel_size = (1, 7)
            padding = (0, 3)
        else:
            kernel_size = (7, 1)
            padding = (3, 0)
        self.axis_mixer = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding, groups=channels, bias=False),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
        )

        hidden_dim = max(8, int(channels * mlp_ratio))
        self.ffn_norm = nn.GroupNorm(4, channels)
        self.ffn = nn.Sequential(
            nn.Conv2d(channels, hidden_dim, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv2d(hidden_dim, channels, kernel_size=1, bias=False),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
        )

    def _cross_attend_to_ll(self, query_tokens, ll_tokens):
        # query_tokens: [B, L, C], ll_tokens: [B, L, C]
        batch_size, num_query_tokens, channels = query_tokens.shape
        num_memory_tokens = ll_tokens.shape[1]

        q = self.q_proj(self.q_norm(query_tokens))
        k, v = self.kv_proj(self.kv_norm(ll_tokens)).chunk(2, dim=-1)

        # [B, heads, tokens, head_dim], compatible with PyTorch SDPA/Flash kernels.
        q = q.view(batch_size, num_query_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, num_memory_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, num_memory_tokens, self.num_heads, self.head_dim).transpose(1, 2)

        attended = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.attn_dropout if self.training else 0.0,
        )
        attended = attended.transpose(1, 2).reshape(batch_size, num_query_tokens, channels)
        return self.proj_dropout(self.out_proj(attended))

    @staticmethod
    def _scatter_latents(latent_updates, assignment, height, width):
        # latent_updates: [B, L, C], assignment: [B, L, HW]
        # dense_updates: [B, C, H, W]
        dense_updates = torch.bmm(assignment.transpose(1, 2), latent_updates)
        return dense_updates.transpose(1, 2).reshape(-1, latent_updates.shape[-1], height, width)

    def forward(self, x):
        # x: [B, 4*C, H, W], packed as [HH, HL, LH, LL]
        batch_size, channels, height, width = x.shape
        if channels != self.channels:
            raise ValueError(f"Expected {self.channels} channels, but got {channels}.")

        hh, hl, lh, ll = torch.chunk(x, 4, dim=1)  # each: [B, C, H, W]

        # 1) Local directional feature extraction, still no cross-subband mixing.
        hh = self.hh_encoder(hh)
        hl = self.hl_encoder(hl)
        lh = self.lh_encoder(lh)
        ll = self.ll_encoder(ll)

        # 2) Lightweight dynamic frequency gating for noisy radar robustness.
        hh, hl, lh, ll = self.freq_gate(hh, hl, lh, ll)

        # 3) TokenLearner compression: [B, C, H, W] -> [B, L, C].
        hh_latent, hh_assign, _ = self.hh_tokens(hh)
        hl_latent, hl_assign, _ = self.hl_tokens(hl)
        lh_latent, lh_assign, _ = self.lh_tokens(lh)
        ll_latent, _, _ = self.ll_tokens(ll)

        # 4) Latent cross attention. High-frequency bands query LL semantic memory.
        hh_update = self._cross_attend_to_ll(hh_latent, ll_latent)
        hl_update = self._cross_attend_to_ll(hl_latent, ll_latent)
        lh_update = self._cross_attend_to_ll(lh_latent, ll_latent)

        scale = self.residual_scale.to(dtype=x.dtype)
        hh = hh + scale * self._scatter_latents(hh_update, hh_assign, height, width)
        hl = hl + scale * self._scatter_latents(hl_update, hl_assign, height, width)
        lh = lh + scale * self._scatter_latents(lh_update, lh_assign, height, width)

        # LL remains the stable low-frequency prior; refine it only locally to
        # avoid over-writing reconstruction-critical low-frequency energy.
        fused = torch.cat([hh, hl, lh, ll], dim=1)  # [B, 4*C, H, W]

        # 5) Efficient row/column long-range mixing + channel FFN in NCHW layout.
        fused = fused + scale * self.axis_mixer(fused)
        fused = fused + scale * self.ffn(self.ffn_norm(fused))
        return fused


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
            self.high_freq_fuse = nn.Conv2d(in_channel * 3, in_channel * 3, kernel_size=1)
            self.subband_fuse_row = WaveletSubbandMHSA2D(
                channels=subband_channels,
                num_heads=num_heads,
                attention_axis="row",
                patch_size=attention_patch_size,
            )
            self.subband_fuse_col = WaveletSubbandMHSA2D(
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
        self.wavelet_scale = nn.Parameter(torch.tensor(0.1))
        self.residual_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        wavelet = self.wavelet_enhance.dwt2d(x, split_subbands=False)
        if self.subband_fuse_type == "mhsa":
            hh, hl, lh, ll = torch.chunk(wavelet, 4, dim=1)
            high_freq = torch.cat([hh, hl, lh], dim=1)
            high_freq = self.high_freq_fuse(high_freq)
            wavelet = torch.cat([high_freq, ll], dim=1)
            fused_subbands = self.subband_fuse_row(wavelet)
            fused_subbands = self.subband_fuse_col(fused_subbands)
        elif self.subband_fuse_type == "conv1x1":
            fused_subbands = self.subband_fuse(wavelet)
        identity = self.wavelet_enhance.idwt2d(fused_subbands)
        identity = self.shortcut_adjust(identity)
        identity_x = self.shortcut(x)
        residual = self.bn2(self.conv2(self.prelu(self.bn1(self.conv1(x)))))
        return (
                identity_x
                + self.wavelet_scale * identity
                + self.residual_scale * residual
        )

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
