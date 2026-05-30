"""3D SRGAN-style ADC generator with the same interface as ``model.Generator_radar3D_adc``.

The generator in ``model.py`` accepts a 5D ADC tensor shaped as
``[batch, input_dim, low_azimuth, range, doppler]`` and returns
``[batch, input_dim, low_azimuth * scale_factor, range, doppler]`` by predicting
``input_dim * scale_factor`` channels and reshaping those channels into the
azimuth dimension.  This module keeps that contract while using a deeper
SRGAN-style residual trunk for experiments that need a 3D SRGAN baseline.
"""

import torch
from torch import nn


class SRGANResidualBlock3D(nn.Module):
    """Residual block used by the 3D SRGAN generator trunk."""

    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(channels),
            nn.PReLU(),
            nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(channels),
        )

    def forward(self, x):
        return x + self.block(x)


class AzimuthPixelShuffle3D(nn.Module):
    """Move predicted scale channels into the azimuth dimension.

    This layer intentionally mirrors the final reshape used by
    ``model.Generator_radar3D_adc`` so the generator can be swapped into the
    existing training and inference code without changing tensor shapes.
    """

    def __init__(self, scale_factor, input_dim):
        super().__init__()
        if int(scale_factor) != scale_factor:
            raise ValueError("scale_factor must be an integer because channels are reshaped into azimuth bins")
        self.scale_factor = int(scale_factor)
        self.input_dim = input_dim

    def forward(self, x, input_shape):
        batch_size, input_channels, low_azimuth, range_size, doppler_size = input_shape
        expected_channels = self.input_dim * self.scale_factor
        if x.shape[1] != expected_channels:
            raise ValueError(f"Expected {expected_channels} channels before shuffle, but got {x.shape[1]}")
        if input_channels != self.input_dim:
            raise ValueError(f"Expected input_dim={self.input_dim}, but got {input_channels} input channels")

        return x.contiguous().view(
            batch_size,
            input_channels,
            low_azimuth * self.scale_factor,
            range_size,
            doppler_size,
        )


class Generator_radar3D_adc_SRGAN(nn.Module):
    """3D SRGAN generator matching ``model.Generator_radar3D_adc`` I/O.

    Args:
        scale_factor: Azimuth upsampling factor. The output azimuth dimension is
            ``input_azimuth * scale_factor``.
        input_dim: Number of ADC channels in the input tensor.
        base_channels: Width of the SRGAN feature trunk.
        num_residual_blocks: Number of residual blocks in the SRGAN trunk.

    Input shape:
        ``[batch, input_dim, low_azimuth, range, doppler]``

    Output shape:
        ``[batch, input_dim, low_azimuth * scale_factor, range, doppler]``
    """

    def __init__(self, scale_factor, input_dim=1, base_channels=64, num_residual_blocks=8):
        super().__init__()
        if int(scale_factor) != scale_factor:
            raise ValueError("scale_factor must be an integer because channels are reshaped into azimuth bins")

        self.input_dim = input_dim
        self.scale_factor = int(scale_factor)
        out_dim = input_dim * self.scale_factor

        self.head = nn.Sequential(
            nn.Conv3d(input_dim, base_channels, kernel_size=9, padding=4),
            nn.PReLU(),
        )
        self.residual_trunk = nn.Sequential(
            *[SRGANResidualBlock3D(base_channels) for _ in range(num_residual_blocks)]
        )
        self.trunk_tail = nn.Sequential(
            nn.Conv3d(base_channels, base_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(base_channels),
        )
        self.reconstruction = nn.Conv3d(base_channels, out_dim, kernel_size=9, padding=4)
        self.azimuth_shuffle = AzimuthPixelShuffle3D(self.scale_factor, input_dim)

    def forward(self, x):
        input_shape = x.shape
        features = self.head(x)
        trunk = self.trunk_tail(self.residual_trunk(features))
        out = self.reconstruction(features + trunk)
        return self.azimuth_shuffle(out, input_shape)


class Discriminator_radar3D_adc_SRGAN(nn.Module):
    """Patch-style 3D discriminator for SRGAN ADC experiments.

    The discriminator consumes high-resolution ADC tensors shaped like the
    generator output, ``[batch, input_dim, azimuth, range, doppler]``, and emits
    one real/fake logit per sample.
    """

    def __init__(self, input_dim=1, base_channels=32):
        super().__init__()

        def discriminator_block(in_channels, out_channels, stride):
            return nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
                nn.BatchNorm3d(out_channels),
                nn.LeakyReLU(0.2, inplace=True),
            )

        self.features = nn.Sequential(
            nn.Conv3d(input_dim, base_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            discriminator_block(base_channels, base_channels, stride=2),
            discriminator_block(base_channels, base_channels * 2, stride=1),
            discriminator_block(base_channels * 2, base_channels * 2, stride=2),
            discriminator_block(base_channels * 2, base_channels * 4, stride=1),
            discriminator_block(base_channels * 4, base_channels * 4, stride=2),
            discriminator_block(base_channels * 4, base_channels * 8, stride=1),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(base_channels * 8, base_channels * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(base_channels * 4, 1),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# Backward-friendly aliases for convenient imports in experiments.
Generator_radar3D_adc_srgan = Generator_radar3D_adc_SRGAN
Discriminator_radar3D_adc_srgan = Discriminator_radar3D_adc_SRGAN
