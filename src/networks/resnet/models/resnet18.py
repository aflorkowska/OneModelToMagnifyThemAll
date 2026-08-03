"""
Source: https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/CNN_architectures/pytorch_resnet.py
From scratch implementation of the famous ResNet models.
The intuition for ResNet is simple and clear, but to code
it didn't feel super clear at first, even when reading Pytorch own
implementation.

Programmed by Aladdin Persson <aladdin.persson at hotmail dot com>
*    2020-04-12 Initial coding
*    2022-12-20 Update comments, code revision, checked still works with latest PyTorch version

### Resnet 18 and 34 - implementation based on: https://github.com/robmarkcole/resent18-from-scratch/blob/main/resnet18.py
"""
import os
import sys
import torch
import torch.nn as nn
from networks.resnet.models.resnet_utils import (
    BasicBlock,
    ConditionalBasicBlock,
    Sequential,
    SequentialWithCondition,
    ChannelwiseLayerNorm,
    ChannelwiseConditionalLayerNorm,
)

from networks.utils.model_utils import model_summary_with_pixel_size

class ResNet18(nn.Module):
    def __init__(
        self, image_channels, num_classes, variant="base", num_conditions=None
    ):
        super(ResNet18, self).__init__()

        self.variant = variant
        self.in_channels = 64
        self.conv1_out_channels = 64
        self.expansion = 1

        # Validate parameters
        if variant == "conditional_normalization" and num_conditions is None:
            raise ValueError(f"num_conditions is required for variant='{variant}'")

        # First convolutional layer
        if variant == "conditional_normalization":
            self.conv1 = nn.Conv2d(
                image_channels,
                self.conv1_out_channels,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False,
            )
            self.ln1 = ChannelwiseConditionalLayerNorm(
                (self.conv1_out_channels,), num_conditions, self.conv1_out_channels
            )
        else:  # base
            self.conv1 = nn.Conv2d(
                image_channels,
                self.conv1_out_channels,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False,
            )
            self.ln1 = ChannelwiseLayerNorm((self.conv1_out_channels,))

        self.relu = nn.LeakyReLU(0.2)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ResNet layers
        layers = [2, 2, 2, 2]
        if variant == "base":
            self.layer1 = self._make_layer_base(
                BasicBlock, layers[0], intermediate_channels=64, stride=1
            )
            self.layer2 = self._make_layer_base(
                BasicBlock, layers[1], intermediate_channels=128, stride=2
            )
            self.layer3 = self._make_layer_base(
                BasicBlock, layers[2], intermediate_channels=256, stride=2
            )
            self.layer4 = self._make_layer_base(
                BasicBlock, layers[3], intermediate_channels=512, stride=2
            )
        else:  # conditional_normalization
            self.num_conditions = num_conditions
            self.layer1 = self._make_layer_conditional(
                ConditionalBasicBlock, layers[0], intermediate_channels=64, stride=1
            )
            self.layer2 = self._make_layer_conditional(
                ConditionalBasicBlock, layers[1], intermediate_channels=128, stride=2
            )
            self.layer3 = self._make_layer_conditional(
                ConditionalBasicBlock, layers[2], intermediate_channels=256, stride=2
            )
            self.layer4 = self._make_layer_conditional(
                ConditionalBasicBlock, layers[3], intermediate_channels=512, stride=2
            )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * self.expansion, num_classes)

    def forward(self, x, pixel_size=None):
        if self.variant == "base":
            x = self.conv1(x)
            x = self.ln1(x)
        elif self.variant == "conditional_normalization":
            x = self.conv1(x)
            x = self.ln1(x, pixel_size)

        x = self.relu(x)
        x = self.maxpool(x)

        if self.variant == "base":
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
        else:
            x = self.layer1(x, pixel_size)
            x = self.layer2(x, pixel_size)
            x = self.layer3(x, pixel_size)
            x = self.layer4(x, pixel_size)

        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return x

    def _make_layer_base(
        self, block, num_residual_blocks, intermediate_channels, stride
    ):
        downsample = None
        layers = []

        if stride != 1 or self.in_channels != intermediate_channels * self.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels,
                    intermediate_channels * self.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
            )

        layers.append(
            block(
                self.in_channels,
                intermediate_channels,
                stride,
                self.expansion,
                downsample,
            )
        )
        self.in_channels = intermediate_channels * self.expansion

        for i in range(num_residual_blocks - 1):
            layers.append(
                block(self.in_channels, intermediate_channels, expansion=self.expansion)
            )

        return Sequential(*layers)

    def _make_layer_conditional(
        self, block, num_residual_blocks, intermediate_channels, stride
    ):
        downsample = None
        layers = []

        if stride != 1 or self.in_channels != intermediate_channels * self.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels,
                    intermediate_channels * self.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
            )

        layers.append(
            block(
                self.in_channels,
                intermediate_channels,
                self.num_conditions,
                stride,
                self.expansion,
                downsample,
                variant=self.variant,
            )
        )
        self.in_channels = intermediate_channels * self.expansion

        for i in range(num_residual_blocks - 1):
            layers.append(
                block(
                    self.in_channels,
                    intermediate_channels,
                    self.num_conditions,
                    expansion=self.expansion,
                    variant=self.variant,
                )
            )

        return SequentialWithCondition(*layers)
        return SequentialWithCondition(*layers)


from torchinfo import summary

if __name__ == "__main__":
    BATCH_SIZE = 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classes = 2
    input_size = (3, 224, 224)

    # Test base variant
    print("Testing base variant...")
    input_tensor = torch.randn(BATCH_SIZE, *input_size).to(device)
    net = ResNet18(image_channels=3, num_classes=classes, variant="base").to(device)
    summary(net, input_size=(BATCH_SIZE, 3, 224, 224), device="cuda")
    y = net(input_tensor)
    print("Output shape", y.size())
    assert y.size() == torch.Size([BATCH_SIZE, classes])

    # Test conditional_normalization variant
    print("\nTesting conditional_normalization variant...")
    pixel_size = torch.stack(
        [
            torch.full((BATCH_SIZE,), 0.5, device=device),
            torch.full((BATCH_SIZE,), 0.5, device=device),
        ],
        dim=1,
    )
    net = ResNet18(
        image_channels=3,
        num_classes=classes,
        variant="conditional_normalization",
        num_conditions=2,
    ).to(device)
    y = net(input_tensor, pixel_size)
    model_summary_with_pixel_size(net, input_size=(BATCH_SIZE, 3, 224, 224), pixel_size=pixel_size)
    print("Output shape", y.size())
    assert y.size() == torch.Size([BATCH_SIZE, classes])
