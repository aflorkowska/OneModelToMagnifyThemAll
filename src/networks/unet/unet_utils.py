"""Parts of the U-Net model - Unified implementation supporting all variants"""

"""
    Paper: `U-Net: Convolutional Networks for Biomedical Image Segmentation
    <https://arxiv.org/abs/1505.04597>`_

    Paper authors: Olaf Ronneberger, Philipp Fischer, Thomas Brox

    Implemented by https://github.com/milesial/Pytorch-UNet/tree/master
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm(nn.Module):
    """Standard LayerNorm implementation"""

    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super(LayerNorm, self).__init__()
        self.eps = eps
        self.normalized_shape = (
            normalized_shape
            if isinstance(normalized_shape, (tuple, list))
            else (normalized_shape,)
        )
        self.elementwise_affine = elementwise_affine

        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(*self.normalized_shape))
            self.bias = nn.Parameter(torch.zeros(*self.normalized_shape))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x):
        # x: (B, ..., *normalized_shape), e.g. (B, C, H, W)
        dims = tuple(range(-len(self.normalized_shape), 0))
        mean = x.mean(dim=dims, keepdim=True)
        var = x.var(dim=dims, keepdim=True, unbiased=False)

        x_normalized = (x - mean) / torch.sqrt(var + self.eps)

        if self.elementwise_affine:
            return x_normalized * self.weight + self.bias
        else:
            return x_normalized


class ChannelwiseLayerNorm(nn.Module):
    """LayerNorm that operates on channel dimension for 2D inputs (B, C, H, W)"""

    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super().__init__()
        self.ln = LayerNorm(normalized_shape, eps, elementwise_affine)

    def forward(self, x):
        # x: (B, C, H, W)
        x = x.permute(0, 2, 3, 1)  # (B, H, W, C)
        x = self.ln(x)
        x = x.permute(0, 3, 1, 2)  # (B, C, H, W)
        return x


class ChannelwiseConditionalLayerNorm(nn.Module):
    """Conditional LayerNorm for channel-wise normalization"""

    def __init__(
        self,
        normalized_shape,
        num_conditions,
        hidden_dim=128,
        eps=1e-5,
        elementwise_affine=True,
    ):
        super().__init__()
        self.ln = ConditionalLayerNorm(
            normalized_shape, num_conditions, hidden_dim, eps, elementwise_affine
        )

    def forward(self, x, condition):
        # x: (B, C, H, W)
        x = x.permute(0, 2, 3, 1)  # (B, H, W, C)
        x = self.ln(x, condition)
        x = x.permute(0, 3, 1, 2)  # (B, C, H, W)
        return x


class ConditionalLayerNorm(nn.Module):
    """LayerNorm with condition-based gamma/beta generation via MLP"""

    def __init__(
        self,
        normalized_shape,
        num_conditions,
        hidden_dim=128,
        eps=1e-5,
        elementwise_affine=True,
    ):
        super().__init__()
        self.eps = eps
        self.normalized_shape = (
            normalized_shape
            if isinstance(normalized_shape, (tuple, list))
            else (normalized_shape,)
        )
        self.elementwise_affine = elementwise_affine

        # MLP to generate weight and bias based on condition
        if self.elementwise_affine:
            self.condition_fc1 = nn.Linear(num_conditions, hidden_dim)
            self.condition_fc2 = nn.Linear(hidden_dim, hidden_dim)
            self.condition_fc3 = nn.Linear(hidden_dim, hidden_dim)
            self.weight_generator = nn.Linear(
                hidden_dim, int(torch.prod(torch.tensor(self.normalized_shape)))
            )
            self.bias_generator = nn.Linear(
                hidden_dim, int(torch.prod(torch.tensor(self.normalized_shape)))
            )

            self.act = nn.LeakyReLU(0.01)

            # Init weight/bias generator to produce γ≈1, β≈0 at the start
            nn.init.zeros_(self.weight_generator.weight)
            nn.init.ones_(self.weight_generator.bias)  # γ start ≈ 1
            nn.init.zeros_(self.bias_generator.weight)
            nn.init.zeros_(self.bias_generator.bias)  # β start ≈ 0
        else:
            self.condition_fc1 = None
            self.condition_fc2 = None
            self.condition_fc3 = None
            self.weight_generator = None
            self.bias_generator = None

    def forward(self, x, condition):
        dims = tuple(range(-len(self.normalized_shape), 0))
        mean = x.mean(dim=dims, keepdim=True)
        var = x.var(dim=dims, keepdim=True, unbiased=False)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)

        if not self.elementwise_affine:
            return x_norm

        cond = self.act(self.condition_fc1(condition))
        cond = self.act(self.condition_fc2(cond))
        cond = self.act(self.condition_fc3(cond))

        weight = self.weight_generator(cond).view(x.shape[0], *self.normalized_shape)
        bias = self.bias_generator(cond).view(x.shape[0], *self.normalized_shape)

        while weight.dim() < x.dim():
            weight = weight.unsqueeze(1)
            bias = bias.unsqueeze(1)

        return weight * x_norm + bias


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels

        self.c1 = nn.Conv2d(
            in_channels, mid_channels, kernel_size=3, padding=1, bias=False
        )
        self.ln1 = ChannelwiseLayerNorm((mid_channels,))
        self.relu1 = nn.ReLU(inplace=True)
        self.c2 = nn.Conv2d(
            mid_channels, out_channels, kernel_size=3, padding=1, bias=False
        )
        self.ln2 = ChannelwiseLayerNorm((out_channels,))
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.c1(x)
        x = self.ln1(x)
        x = self.relu1(x)
        x = self.c2(x)
        x = self.ln2(x)
        x = self.relu2(x)
        return x


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x):
        x = self.maxpool(x)
        return self.conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels, in_channels // 2, kernel_size=2, stride=2
            )
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """Output convolution layer"""

    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class ConditionalNormDoubleConv(nn.Module):
    """(convolution => [CondLayerNorm] => ReLU) * 2 with conditional normalization"""

    def __init__(self, num_conditions, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels

        self.conv1 = nn.Conv2d(
            in_channels, mid_channels, kernel_size=3, padding=1, bias=False
        )
        self.norm1 = ChannelwiseConditionalLayerNorm(
            (mid_channels,), num_conditions, mid_channels
        )
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(
            mid_channels, out_channels, kernel_size=3, padding=1, bias=False
        )
        self.norm2 = ChannelwiseConditionalLayerNorm(
            (out_channels,), num_conditions, out_channels
        )
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x, pixel_size):
        x = self.conv1(x)
        x = self.norm1(x, pixel_size)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.norm2(x, pixel_size)
        x = self.relu2(x)

        return x


class ConditionalNormDown(nn.Module):
    """Downscaling with maxpool then double conv (conditional normalization)"""

    def __init__(self, num_conditions, in_channels, out_channels):
        super().__init__()
        self.maxpool = nn.MaxPool2d(2)
        self.conv = ConditionalNormDoubleConv(num_conditions, in_channels, out_channels)

    def forward(self, x, pixel_size):
        x = self.maxpool(x)
        return self.conv(x, pixel_size)


class ConditionalNormUp(nn.Module):
    """Upscaling then double conv (conditional normalization)"""

    def __init__(self, num_conditions, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = ConditionalNormDoubleConv(
                num_conditions, in_channels, out_channels, in_channels // 2
            )
        else:
            self.up = nn.ConvTranspose2d(
                in_channels, in_channels // 2, kernel_size=2, stride=2
            )
            self.conv = ConditionalNormDoubleConv(
                num_conditions, in_channels, out_channels
            )

    def forward(self, x1, x2, pixel_size):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x, pixel_size)
