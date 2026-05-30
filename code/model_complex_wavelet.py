import math

import torch
from torch import nn

from complex.complex_wavelets import complex_dwt2d_ra, complex_idwt2d_ra


_SUBBAND_ORDER = ("HH", "HL", "LH", "LL")


class Wavelet_radar3D_adc(nn.Module):
    def __init__(self, scale_factor, input_dim=2):
        super().__init__()

        if input_dim != 2:
            raise ValueError(
                "Wavelet_radar3D_adc expects input_dim=2 because the complex input is split into real/imag channels."
            )

        out_dim = input_dim * float(scale_factor)
        if not math.isclose(out_dim, round(out_dim)):
            raise ValueError(
                f"scale_factor={scale_factor} leads to a non-integer output channel count: {out_dim}"
            )

        self.input_dim = input_dim
        self.scale_factor = float(scale_factor)
        self.out_dim = int(round(out_dim))
        self.num_subbands = len(_SUBBAND_ORDER)

        if self.out_dim % 2 != 0:
            raise ValueError(
                f"Output channels must be even for real/imag pairing, but got {self.out_dim}."
            )

        self.block1 = nn.Sequential(
            nn.Conv3d(input_dim, 8, kernel_size=9, padding=4),
            nn.PReLU(),
        )
        self.block2 = ResidualBlock3D(8)
        self.block3 = ResidualBlock3D(8)
        self.block4 = ResidualBlock3D(8)
        self.block5 = ResidualBlock3D(8)
        self.block6 = ResidualBlock3D(8)
        self.block7 = nn.Sequential(
            nn.Conv3d(8, 8, kernel_size=3, padding=1),
            nn.BatchNorm3d(8),
        )
        self.block8 = nn.Sequential(
            nn.Conv3d(8, self.out_dim, kernel_size=9, padding=4),
        )

    @staticmethod
    def _validate_input(x: torch.Tensor) -> None:
        if x.ndim != 4:
            raise ValueError(f"Expected input shape [B, A, R, D], but got {tuple(x.shape)}")
        if not torch.is_complex(x):
            raise TypeError(f"Expected a complex input tensor, but got dtype={x.dtype}")

        _, azimuth_bins, range_bins, _ = x.shape
        if azimuth_bins % 2 != 0 or range_bins % 2 != 0:
            raise ValueError(
                f"Wavelet transform expects even azimuth/range dimensions, but got A={azimuth_bins}, R={range_bins}."
            )

    @torch.no_grad()
    def _to_channels_last_3d_(self):
        self.to(memory_format=torch.channels_last_3d)
        return self

    def _wavelet_encode(self, x: torch.Tensor):
        batch_size, azimuth_bins, range_bins, doppler_bins = x.shape

        x_ra = x.permute(0, 3, 1, 2).contiguous().reshape(batch_size * doppler_bins, azimuth_bins, range_bins)
        coeffs = complex_dwt2d_ra(x_ra)

        coeff_tensor = torch.cat([coeffs[name] for name in _SUBBAND_ORDER], dim=1)
        coeff_tensor = coeff_tensor.reshape(batch_size, doppler_bins, coeff_tensor.shape[1], coeff_tensor.shape[2])
        coeff_tensor = coeff_tensor.permute(0, 2, 3, 1).contiguous()

        coeff_volume = torch.view_as_real(coeff_tensor).permute(0, 4, 1, 2, 3)
        if coeff_volume.is_cuda:
            coeff_volume = coeff_volume.contiguous(memory_format=torch.channels_last_3d)
        else:
            coeff_volume = coeff_volume.contiguous()

        return coeff_volume, doppler_bins

    def _wavelet_decode(self, coeff_volume: torch.Tensor, doppler_bins: int) -> torch.Tensor:
        batch_size, out_channels, packed_subbands, range_half, current_doppler_bins = coeff_volume.shape
        if current_doppler_bins != doppler_bins:
            raise ValueError(
                f"Doppler dimension mismatch: expected {doppler_bins}, but got {current_doppler_bins}."
            )
        if out_channels % 2 != 0:
            raise ValueError(
                f"Output channel dimension must be even for real/imag pairing, but got {out_channels}."
            )

        coeff_volume = coeff_volume.reshape(
            batch_size,
            2,
            out_channels // 2,
            packed_subbands,
            range_half,
            doppler_bins,
        )
        coeff_volume = coeff_volume.permute(0, 5, 1, 2, 3, 4).contiguous()
        coeff_tensor = coeff_volume.reshape(
            batch_size * doppler_bins,
            2,
            (out_channels // 2) * packed_subbands,
            range_half,
        )

        coeff_complex = torch.complex(coeff_tensor[:, 0], coeff_tensor[:, 1])
        if coeff_complex.shape[1] % self.num_subbands != 0:
            raise ValueError(
                f"Wavelet coefficient channels must be divisible by {self.num_subbands}, but got {coeff_complex.shape[1]}."
            )

        hh, hl, lh, ll = torch.chunk(coeff_complex, self.num_subbands, dim=1)
        recon = complex_idwt2d_ra({"HH": hh, "HL": hl, "LH": lh, "LL": ll})

        recon = recon.reshape(batch_size, doppler_bins, recon.shape[1], recon.shape[2])
        recon = recon.permute(0, 2, 3, 1).contiguous()
        return recon

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._validate_input(x)

        x_wavelet, doppler_bins = self._wavelet_encode(x)

        block1 = self.block1(x_wavelet)
        block2 = self.block2(block1)
        block3 = self.block3(block2)
        block4 = self.block4(block3)
        block5 = self.block5(block4)
        block6 = self.block6(block5)
        block7 = self.block7(block6)
        block7 = block7.add_(block1)
        block8 = self.block8(block7)

        return self._wavelet_decode(block8, doppler_bins)


class ResidualBlock3D(nn.Module):
    def __init__(self, channels=1):
        super().__init__()
        self.conv1 = nn.Conv3d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(channels)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv3d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.conv1(x)
        residual = self.bn1(residual)
        residual = self.prelu(residual)
        residual = self.conv2(residual)
        residual = self.bn2(residual)
        return residual.add_(x)
