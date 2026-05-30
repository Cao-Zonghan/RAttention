"""
Radar Mamba Block (RM Block) recreated from the provided architecture diagram.

The original figure only exposes the high-level data flow, so this module implements
an executable approximation that preserves the main structure:

1. parallel multi-scale depthwise convolutions (1x1 / 3x3 / 5x5)
2. LayerNorm + Linear projection
3. four parallel RHSS branches with learnable branch weights
4. concat + residual add + LayerNorm + Linear projection

The RHSS module follows the diagram with:
- dual linear projections (value / gate)
- depthwise convolution
- hybrid multi-scan fusion (global scan + local scan)
- element-wise gating
- LayerNorm + Linear projection
- residual connection

Input shapes:
- channels_first: [B, C, H, W]
- channels_last:  [B, H, W, C]
"""

import math
from typing import Tuple

import torch
from torch import nn
import torch.nn.functional as F


_SCAN_CONFIGS = (
    (False, False),
    (False, True),
    (True, False),
    (True, True),
)


def _to_channel_last(x: torch.Tensor) -> torch.Tensor:
    return x.permute(0, 2, 3, 1).contiguous()



def _to_channel_first(x: torch.Tensor) -> torch.Tensor:
    return x.permute(0, 3, 1, 2).contiguous()



def _snake_scan_order(
    height: int,
    width: int,
    transpose: bool,
    reverse: bool,
    device: torch.device,
) -> torch.Tensor:
    grid = torch.arange(height * width, device=device).view(height, width)
    if transpose:
        grid = grid.t()

    rows = []
    for row_idx in range(grid.size(0)):
        row = grid[row_idx]
        if row_idx % 2 == 1:
            row = torch.flip(row, dims=(0,))
        rows.append(row)

    order = torch.cat(rows, dim=0)
    if reverse:
        order = torch.flip(order, dims=(0,))
    return order



def _inverse_permutation(order: torch.Tensor) -> torch.Tensor:
    inverse = torch.empty_like(order)
    inverse[order] = torch.arange(order.numel(), device=order.device)
    return inverse



def _pad_to_window(x: torch.Tensor, window_size: int) -> Tuple[torch.Tensor, int, int]:
    if window_size <= 0:
        raise ValueError(f"window_size must be positive, but got {window_size}")

    height, width = x.shape[1], x.shape[2]
    pad_h = (window_size - height % window_size) % window_size
    pad_w = (window_size - width % window_size) % window_size

    if pad_h == 0 and pad_w == 0:
        return x, height, width

    x_cf = _to_channel_first(x)
    x_cf = F.pad(x_cf, (0, pad_w, 0, pad_h))
    return _to_channel_last(x_cf), height, width


class DepthwiseConv2dChannelLast(nn.Module):
    def __init__(self, dim: int, kernel_size: int) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be odd, but got {kernel_size}")

        self.conv = nn.Conv2d(
            dim,
            dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=dim,
            bias=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _to_channel_first(x)
        x = self.conv(x)
        return _to_channel_last(x)


class MultiScaleDepthwiseConv2d(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dw_conv_1 = DepthwiseConv2dChannelLast(dim, kernel_size=1)
        self.dw_conv_3 = DepthwiseConv2dChannelLast(dim, kernel_size=3)
        self.dw_conv_5 = DepthwiseConv2dChannelLast(dim, kernel_size=5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dw_conv_1(x) + self.dw_conv_3(x) + self.dw_conv_5(x)


class ScanMixer1D(nn.Module):
    def __init__(self, dim: int, kernel_size: int) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be odd, but got {kernel_size}")

        self.depthwise = nn.Conv1d(
            dim,
            dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=dim,
            bias=True,
        )
        self.pointwise = nn.Conv1d(dim, dim, kernel_size=1, bias=True)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.act(x)
        x = self.pointwise(x)
        return x


class HybridScan2D(nn.Module):
    def __init__(self, dim: int, window_size: int = 4, scan_kernel_size: int = 7) -> None:
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.global_mixers = nn.ModuleList(
            [ScanMixer1D(dim, scan_kernel_size) for _ in range(len(_SCAN_CONFIGS))]
        )
        self.local_mixers = nn.ModuleList(
            [ScanMixer1D(dim, scan_kernel_size) for _ in range(len(_SCAN_CONFIGS))]
        )
        self.global_scale = nn.Parameter(torch.tensor(1.0))
        self.local_scale = nn.Parameter(torch.tensor(1.0))

    def _apply_scan(
        self,
        x: torch.Tensor,
        order: torch.Tensor,
        mixer: nn.Module,
        height: int,
        width: int,
    ) -> torch.Tensor:
        batch_size, _, _, channels = x.shape
        inverse = _inverse_permutation(order)

        seq = x.reshape(batch_size, height * width, channels)
        seq = seq.index_select(dim=1, index=order)
        seq = seq.transpose(1, 2).contiguous()
        seq = mixer(seq)
        seq = seq.transpose(1, 2).contiguous()
        seq = seq.index_select(dim=1, index=inverse)
        return seq.reshape(batch_size, height, width, channels)

    def _global_scan(self, x: torch.Tensor) -> torch.Tensor:
        height, width = x.shape[1], x.shape[2]
        outputs = []

        for mixer, (transpose, reverse) in zip(self.global_mixers, _SCAN_CONFIGS):
            order = _snake_scan_order(height, width, transpose, reverse, x.device)
            outputs.append(self._apply_scan(x, order, mixer, height, width))

        return torch.stack(outputs, dim=0).mean(dim=0)

    def _local_scan(self, x: torch.Tensor) -> torch.Tensor:
        x_pad, orig_h, orig_w = _pad_to_window(x, self.window_size)
        batch_size, pad_h, pad_w, channels = x_pad.shape
        num_h = pad_h // self.window_size
        num_w = pad_w // self.window_size

        windows = x_pad.view(
            batch_size,
            num_h,
            self.window_size,
            num_w,
            self.window_size,
            channels,
        )
        windows = windows.permute(0, 1, 3, 2, 4, 5).contiguous()
        windows = windows.view(-1, self.window_size, self.window_size, channels)

        outputs = []
        for mixer, (transpose, reverse) in zip(self.local_mixers, _SCAN_CONFIGS):
            order = _snake_scan_order(
                self.window_size,
                self.window_size,
                transpose,
                reverse,
                x.device,
            )
            outputs.append(
                self._apply_scan(windows, order, mixer, self.window_size, self.window_size)
            )

        windows = torch.stack(outputs, dim=0).mean(dim=0)
        windows = windows.view(batch_size, num_h, num_w, self.window_size, self.window_size, channels)
        windows = windows.permute(0, 1, 3, 2, 4, 5).contiguous()
        x_out = windows.view(batch_size, pad_h, pad_w, channels)
        return x_out[:, :orig_h, :orig_w, :]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        global_features = self._global_scan(x)
        local_features = self._local_scan(x)
        return self.global_scale * global_features + self.local_scale * local_features


class RHSSModule(nn.Module):
    def __init__(
        self,
        dim: int,
        expansion: float = 2.0,
        dw_kernel_size: int = 3,
        scan_kernel_size: int = 7,
        window_size: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        hidden_dim = max(1, int(dim * expansion))

        self.value_proj = nn.Linear(dim, hidden_dim)
        self.gate_proj = nn.Linear(dim, hidden_dim)
        self.dw_conv = DepthwiseConv2dChannelLast(hidden_dim, kernel_size=dw_kernel_size)
        self.multi_scan = HybridScan2D(
            hidden_dim,
            window_size=window_size,
            scan_kernel_size=scan_kernel_size,
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        value = self.value_proj(x)
        gate = torch.sigmoid(self.gate_proj(x))

        value = self.dw_conv(value)
        value = self.multi_scan(value)
        value = value * gate
        value = self.norm(value)
        value = self.out_proj(value)
        value = self.dropout(value)

        return residual + value


class RMBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        inner_dim: int = None,
        num_branches: int = 4,
        rhss_expansion: float = 2.0,
        window_size: int = 4,
        dw_kernel_size: int = 3,
        scan_kernel_size: int = 7,
        dropout: float = 0.0,
        data_format: str = "channels_first",
    ) -> None:
        super().__init__()
        if data_format not in {"channels_first", "channels_last"}:
            raise ValueError(
                f"data_format must be 'channels_first' or 'channels_last', but got {data_format}"
            )

        if inner_dim is None:
            inner_dim = int(math.ceil(dim / num_branches) * num_branches)

        if inner_dim % num_branches != 0:
            raise ValueError(
                f"inner_dim ({inner_dim}) must be divisible by num_branches ({num_branches})"
            )

        branch_dim = inner_dim // num_branches

        self.dim = dim
        self.inner_dim = inner_dim
        self.num_branches = num_branches
        self.data_format = data_format

        self.multi_scale_dw = MultiScaleDepthwiseConv2d(dim)
        self.pre_norm = nn.LayerNorm(dim)
        self.pre_proj = nn.Linear(dim, inner_dim)
        self.branches = nn.ModuleList(
            [
                RHSSModule(
                    branch_dim,
                    expansion=rhss_expansion,
                    dw_kernel_size=dw_kernel_size,
                    scan_kernel_size=scan_kernel_size,
                    window_size=window_size,
                    dropout=dropout,
                )
                for _ in range(num_branches)
            ]
        )
        self.branch_weights = nn.Parameter(torch.ones(num_branches, branch_dim))
        self.post_norm = nn.LayerNorm(inner_dim)
        self.post_proj = nn.Linear(inner_dim, dim)
        self.dropout = nn.Dropout(dropout)

    def _normalize_layout(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"RMBlock expects a 4D tensor, but got shape={tuple(x.shape)}")

        if self.data_format == "channels_first":
            if x.size(1) != self.dim:
                raise ValueError(
                    f"Expected channel dimension {self.dim} at dim=1, but got shape={tuple(x.shape)}"
                )
            return _to_channel_last(x)

        if x.size(-1) != self.dim:
            raise ValueError(
                f"Expected channel dimension {self.dim} at the last axis, but got shape={tuple(x.shape)}"
            )
        return x

    def _restore_layout(self, x: torch.Tensor) -> torch.Tensor:
        if self.data_format == "channels_first":
            return _to_channel_first(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._normalize_layout(x)

        x = self.multi_scale_dw(x)
        x = self.pre_norm(x)
        x = self.pre_proj(x)

        shortcut = x
        branches = torch.chunk(x, self.num_branches, dim=-1)
        outputs = []

        for branch_idx, (branch, branch_input) in enumerate(zip(self.branches, branches)):
            branch_output = branch(branch_input)
            branch_weight = self.branch_weights[branch_idx].view(1, 1, 1, -1)
            outputs.append(branch_output * branch_weight)

        x = torch.cat(outputs, dim=-1)
        x = x + shortcut
        x = self.post_norm(x)
        x = self.post_proj(x)
        x = self.dropout(x)

        return self._restore_layout(x)

__all__ = [
    "DepthwiseConv2dChannelLast",
    "MultiScaleDepthwiseConv2d",
    "ScanMixer1D",
    "HybridScan2D",
    "RHSSModule",
    "RMBlock",
]
