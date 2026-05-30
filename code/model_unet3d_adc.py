import torch
from torch import nn
import torch.nn.functional as F


class ConvBlock3D(nn.Module):
    """Two 3D convolutions used by the 3D U-Net encoder/decoder."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.PReLU(),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.PReLU(),
        )

    def forward(self, x):
        return self.block(x)


class UpBlock3D(nn.Module):
    """Upsample, concatenate the skip feature, then refine with 3D convolutions."""

    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.conv = ConvBlock3D(in_channels + skip_channels, out_channels)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[2:], mode="trilinear", align_corners=False)
        x = torch.cat((skip, x), dim=1)
        return self.conv(x)


class Generator_radar3D_adc_UNet(nn.Module):
    """
    3D U-Net baseline with the same input/output interface as model.Generator_radar3D_adc.

    Input shape:
        [batch, input_dim, low_azimuth, range, doppler]

    Output shape:
        [batch, input_dim, low_azimuth * scale_factor, range, doppler]

    The original Generator_radar3D_adc first predicts input_dim * scale_factor channels
    and then reshapes those channels into the azimuth dimension. This implementation keeps
    that behavior so it can be used as a drop-in comparison model in the existing training
    and inference code.
    """

    def __init__(self, scale_factor, input_dim=2, base_channels=16):
        super().__init__()
        if int(scale_factor) != scale_factor:
            raise ValueError("scale_factor must be an integer because channels are reshaped into azimuth bins")

        self.input_dim = input_dim
        self.scale_factor = int(scale_factor)
        out_dim = input_dim * self.scale_factor

        # Pool only range/doppler dimensions. This keeps the often-small azimuth dimension
        # valid while 3D convolutions still model azimuth-range-doppler context.
        self.pool = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.enc1 = ConvBlock3D(input_dim, base_channels)
        self.enc2 = ConvBlock3D(base_channels, base_channels * 2)
        self.enc3 = ConvBlock3D(base_channels * 2, base_channels * 4)
        self.bottleneck = ConvBlock3D(base_channels * 4, base_channels * 8)

        self.up3 = UpBlock3D(base_channels * 8, base_channels * 4, base_channels * 4)
        self.up2 = UpBlock3D(base_channels * 4, base_channels * 2, base_channels * 2)
        self.up1 = UpBlock3D(base_channels * 2, base_channels, base_channels)

        self.out_conv = nn.Conv3d(base_channels, out_dim, kernel_size=1)

    def forward(self, x):
        shape_buffer = x.shape

        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        bottleneck = self.bottleneck(self.pool(enc3))

        dec3 = self.up3(bottleneck, enc3)
        dec2 = self.up2(dec3, enc2)
        dec1 = self.up1(dec2, enc1)

        out = self.out_conv(dec1)
        out = out.view(
            shape_buffer[0],
            shape_buffer[1],
            int(shape_buffer[2] * self.scale_factor),
            shape_buffer[3],
            shape_buffer[4],
        )
        return out


class ConvBlock2D(nn.Module):
    """Two 2D convolutions used by the 2D U-Net encoder/decoder."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.PReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.PReLU(),
        )

    def forward(self, x):
        return self.block(x)


class UpBlock2D(nn.Module):
    """Upsample, concatenate the skip feature, then refine with 2D convolutions."""

    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.conv = ConvBlock2D(in_channels + skip_channels, out_channels)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat((skip, x), dim=1)
        return self.conv(x)


class Generator_radar2D_adc_UNet(nn.Module):
    """
    2D U-Net baseline with the same 5D input/output interface as Generator_radar3D_adc.

    Input shape:
        [batch, input_dim, low_azimuth, range, doppler]

    Output shape:
        [batch, input_dim, low_azimuth * scale_factor, range, doppler]

    The azimuth dimension is folded into channels before the 2D U-Net, so the network uses
    2D convolutions on range-doppler maps while still producing the same azimuth-upsampled
    tensor shape as the original 3D ADC generator.
    """

    def __init__(self, scale_factor, input_dim=2, base_channels=16, low_azimuth=4):
        super().__init__()
        if int(scale_factor) != scale_factor:
            raise ValueError("scale_factor must be an integer because channels are reshaped into azimuth bins")

        self.input_dim = input_dim
        self.scale_factor = int(scale_factor)
        self.low_azimuth = low_azimuth

        self.enc1 = ConvBlock2D(input_dim * low_azimuth, base_channels)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.enc2 = ConvBlock2D(base_channels, base_channels * 2)
        self.enc3 = ConvBlock2D(base_channels * 2, base_channels * 4)
        self.bottleneck = ConvBlock2D(base_channels * 4, base_channels * 8)

        self.up3 = UpBlock2D(base_channels * 8, base_channels * 4, base_channels * 4)
        self.up2 = UpBlock2D(base_channels * 4, base_channels * 2, base_channels * 2)
        self.up1 = UpBlock2D(base_channels * 2, base_channels, base_channels)

        self.out_conv = nn.Conv2d(base_channels, input_dim * low_azimuth * self.scale_factor, kernel_size=1)

    def forward(self, x):
        batch_size, channels, low_azimuth, range_size, doppler_size = x.shape
        if channels != self.input_dim:
            raise ValueError(f"Expected input_dim={self.input_dim}, but got {channels} input channels")
        if low_azimuth != self.low_azimuth:
            raise ValueError(f"Expected low_azimuth={self.low_azimuth}, but got {low_azimuth}")

        # Fold azimuth into channels: [B, C, A, R, D] -> [B, C*A, R, D].
        x_2d = x.reshape(batch_size, channels * low_azimuth, range_size, doppler_size)

        enc1 = self.enc1(x_2d)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        bottleneck = self.bottleneck(self.pool(enc3))

        dec3 = self.up3(bottleneck, enc3)
        dec2 = self.up2(dec3, enc2)
        dec1 = self.up1(dec2, enc1)

        out = self.out_conv(dec1)
        out = out.view(
            batch_size,
            channels,
            low_azimuth * self.scale_factor,
            range_size,
            doppler_size,
        )
        return out


# Backward-friendly aliases for convenient imports in experiments.
UNet_3D_ADC = Generator_radar3D_adc_UNet
Generator_radar3D_adc_unet = Generator_radar3D_adc_UNet
UNet_2D_ADC = Generator_radar2D_adc_UNet
Generator_radar2D_adc_unet = Generator_radar2D_adc_UNet
