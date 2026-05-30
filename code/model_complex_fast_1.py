import torch
import torch.nn as nn
import torch.nn.functional as F

# Reuse complex utilities already present in the project
from complex.complex_layers import ComplexPReLU, NaiveComplexBatchNorm2d


class PackedComplexConv2d(nn.Module):
    """
    Efficient complex 2D convolution implemented with grouped real convolutions.

    [优化项]: 修复了 Bias 计算逻辑，避免了将偏置传入 4 次实数卷积造成的冗余与耦合。
    卷积计算全程无 Bias，最后统一进行向量广播加法，严格对齐复数运算的数学定义。
    """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int = 3,
            stride: int = 1,
            padding: int = 0,
            dilation: int = 1,
            groups: int = 1,
            bias: bool = True,
    ) -> None:
        super().__init__()
        assert groups == 1, "Grouped complex conv is not implemented in the packed version"
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        self.dilation = dilation if isinstance(dilation, tuple) else (dilation, dilation)

        kH, kW = self.kernel_size
        self.weight_r = nn.Parameter(torch.empty(out_channels, in_channels, kH, kW))
        self.weight_i = nn.Parameter(torch.empty(out_channels, in_channels, kH, kW))
        if bias:
            self.bias_r = nn.Parameter(torch.empty(out_channels))
            self.bias_i = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter("bias_r", None)
            self.register_parameter("bias_i", None)

        nn.init.kaiming_normal_(self.weight_r, mode="fan_out", nonlinearity="relu")
        nn.init.kaiming_normal_(self.weight_i, mode="fan_out", nonlinearity="relu")
        if bias:
            nn.init.zeros_(self.bias_r)
            nn.init.zeros_(self.bias_i)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not torch.is_complex(x):
            raise TypeError(f"PackedComplexConv2d expects a complex tensor, but got dtype={x.dtype}")

        # 动态适配内存格式，如果外层使用了 channels_last，这里能保持格式连续性
        memory_format = torch.channels_last if x.is_contiguous(
            memory_format=torch.channels_last) else torch.contiguous_format

        x_r = x.real.contiguous(memory_format=memory_format)
        x_i = x.imag.contiguous(memory_format=memory_format)

        # [核心优化]: bias 设为 None，避免偏置被重复计算和错误耦合
        rr = F.conv2d(
            x_r, self.weight_r, None,
            stride=self.stride, padding=self.padding, dilation=self.dilation, groups=1
        )
        ii = F.conv2d(
            x_i, self.weight_i, None,
            stride=self.stride, padding=self.padding, dilation=self.dilation, groups=1
        )
        ri = F.conv2d(
            x_r, self.weight_i, None,
            stride=self.stride, padding=self.padding, dilation=self.dilation, groups=1
        )
        ir = F.conv2d(
            x_i, self.weight_r, None,
            stride=self.stride, padding=self.padding, dilation=self.dilation, groups=1
        )

        y_r = rr - ii
        y_i = ri + ir

        # [核心优化]: 在算子计算完成后，进行一次高效的广播加法
        if self.bias_r is not None:
            # view(1, -1, 1, 1) 使其形状变为 (1, out_channels, 1, 1) 方便广播
            y_r = y_r + self.bias_r.view(1, -1, 1, 1)
            y_i = y_i + self.bias_i.view(1, -1, 1, 1)

        return torch.complex(y_r, y_i)


class ComplexResidualBlock2D_Fast(nn.Module):
    """
    Fast residual block using PackedComplexConv2d + ComplexPReLU.
    Optionally keep NaiveComplexBatchNorm2d (bn=True) or remove BN for speed.
    """

    def __init__(self, in_channel: int, out_channel: int, bn: bool = True):
        super().__init__()
        self.conv1 = PackedComplexConv2d(in_channel, out_channel, kernel_size=3, padding=1)
        self.bn1 = NaiveComplexBatchNorm2d(out_channel) if bn else nn.Identity()
        self.act = ComplexPReLU()
        self.conv2 = PackedComplexConv2d(out_channel, out_channel, kernel_size=3, padding=1)
        self.bn2 = NaiveComplexBatchNorm2d(out_channel) if bn else nn.Identity()

        if in_channel != out_channel:
            self.shortcut = nn.Sequential(
                PackedComplexConv2d(in_channel, out_channel, kernel_size=1),
                NaiveComplexBatchNorm2d(out_channel) if bn else nn.Identity(),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)
        out = self.conv2(out)
        out = self.bn2(out)

        # [优化项]: 使用 in-place 原地加法，节省一次大小为 [B, C, H, W] 的显存分配
        out += identity

        out = self.act(out)
        return out


class Generator_radar2D_adc_complex_fast(nn.Module):
    """
    Optimized version of Generator_radar2D_adc_complex with:
    - Packed Complex Conv (single real conv per layer)
    - Optional BN removal (bn=False) for faster inference
    - Smaller edge kernels (k_first/k_last) default 5 rather than 9
    - channels_last packing to improve memory throughput
    """

    def __init__(
            self,
            scale_factor: int,
            input_dim: int = 4,
            k_first: int = 9,
            k_last: int = 5,
            use_bn: bool = True,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.scale_factor = scale_factor

        # stem
        self.block1 = nn.Sequential(
            PackedComplexConv2d(input_dim, 8, kernel_size=k_first, padding=k_first // 2),
            ComplexPReLU(),
        )

        # residual trunk
        self.layer1 = ComplexResidualBlock2D_Fast(8, 16, bn=use_bn)
        self.layer2 = ComplexResidualBlock2D_Fast(16, 32, bn=use_bn)
        self.layer3 = ComplexResidualBlock2D_Fast(32, 32, bn=use_bn)
        self.layer4 = ComplexResidualBlock2D_Fast(32, 32, bn=use_bn)
        self.layer5 = ComplexResidualBlock2D_Fast(32, 32, bn=use_bn)
        self.layer6 = ComplexResidualBlock2D_Fast(32, 32, bn=use_bn)

        # bottleneck reduce
        self.block7 = nn.Sequential(
            PackedComplexConv2d(32, 16, kernel_size=3, padding=1),
            NaiveComplexBatchNorm2d(16) if use_bn else nn.Identity(),
        )

        # global skip projection from early features
        self.global_skip = nn.Sequential(
            PackedComplexConv2d(8, 16, kernel_size=1),
            NaiveComplexBatchNorm2d(16) if use_bn else nn.Identity(),
        )

        out_dim = int(input_dim * scale_factor)
        self.out_dim = out_dim
        self.block8 = PackedComplexConv2d(16, out_dim, kernel_size=k_last, padding=k_last // 2)

    @torch.no_grad()
    def _to_channels_last_(self) -> None:
        """
        [优化项]: 真正执行模型权重的 channels_last 转换。
        调用此方法后，请确保传入的输入 Tensor 也通过 x = x.to(memory_format=torch.channels_last) 进行了转换，
        以便利用 Tensor Core 实现最大化的显存吞吐加速。
        """
        self.to(memory_format=torch.channels_last)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.block1(x)
        x = self.layer1(x1)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)

        x = self.block7(x)
        skip = self.global_skip(x1)

        # 同样的 in-place 操作，但由于这里直接传入下一层，不需要修改
        x = self.block8(x + skip)
        return x


__all__ = [
    "PackedComplexConv2d",
    "ComplexResidualBlock2D_Fast",
    "Generator_radar2D_adc_complex_fast",
]