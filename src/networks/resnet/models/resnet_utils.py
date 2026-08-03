import torch
import torch.nn as nn

class LayerNorm(nn.Module):
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
            self.weight = nn.Parameter(
                torch.ones(*self.normalized_shape, dtype=torch.float32)
            )
            self.bias = nn.Parameter(
                torch.zeros(*self.normalized_shape, dtype=torch.float32)
            )
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

        # MLP to generate weight i bias including condition
        if self.elementwise_affine:
            self.condition_fc1 = nn.Linear(num_conditions, hidden_dim)
            self.condition_fc2 = nn.Linear(hidden_dim, hidden_dim)
            self.condition_fc3 = nn.Linear(hidden_dim, hidden_dim)
            # Improve this network
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
        # x: (B, ..., *normalized_shape), e.g. (B, C, H, W)
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

        # broadcast gamma/beta
        while weight.dim() < x.dim():
            weight = weight.unsqueeze(1)
            bias = bias.unsqueeze(1)

        return weight * x_norm + bias

class BasicBlock(nn.Module):
    """Basic Block for ResNet18 and ResNet34"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        expansion: int = 1,
        downsample: nn.Module = None,
    ) -> None:
        super(BasicBlock, self).__init__()
        self.expansion = expansion
        self.downsample = downsample
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )

        self.ln1 = ChannelwiseLayerNorm((out_channels,))
        self.relu = nn.LeakyReLU(0.2)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels * self.expansion,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.ln2 = ChannelwiseLayerNorm((out_channels * self.expansion,))
        self.ln3 = ChannelwiseLayerNorm((out_channels * self.expansion,))

    def forward(self, x):
        identity = x.clone()
        x = self.conv1(x)
        x = self.ln1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.ln2(x)

        if self.downsample is not None:
            identity = self.downsample(identity)
            identity = self.ln3(identity)

        x = x + identity
        x = self.relu(x)
        return x


class ConditionalBasicBlock(nn.Module):
    """Basic Block for ResNet18 and ResNet34 with Conditional Normalization"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_conditions,
        stride: int = 1,
        expansion: int = 1,
        downsample: nn.Module = None,
        variant: str = "conditional_normalization",
    ) -> None:
        super(ConditionalBasicBlock, self).__init__()
        self.expansion = expansion
        self.downsample = downsample
        self.variant = variant

        if variant == "conditional_normalization":
            # Use standard Conv with Conditional LayerNorm
            self.conv1 = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
            )
            self.conv2 = nn.Conv2d(
                out_channels,
                out_channels * self.expansion,
                kernel_size=3,
                padding=1,
                bias=False,
            )
            num_of_features = 64 if out_channels * expansion <= 256 else 128
            self.ln1 = ChannelwiseConditionalLayerNorm(
                (out_channels,), num_conditions, num_of_features
            )
            self.ln2 = ChannelwiseConditionalLayerNorm(
                (out_channels * self.expansion,), num_conditions, num_of_features
            )
            self.ln3 = ChannelwiseConditionalLayerNorm(
                (out_channels * self.expansion,), num_conditions, num_of_features
            )
        else:
            raise ValueError(f"Unknown variant: {variant}")

        self.relu = nn.LeakyReLU(0.2)

    def forward(self, x, pixel_size):
        identity = x.clone()

        if self.variant == "conditional_normalization":
            x = self.conv1(x)
            x = self.ln1(x, pixel_size)
            x = self.relu(x)
            x = self.conv2(x)
            x = self.ln2(x, pixel_size)

            if self.downsample is not None:
                identity = self.downsample(identity)
                identity = self.ln3(identity, pixel_size)

        x = x + identity
        x = self.relu(x)
        return x


class Sequential(nn.Sequential):
    def forward(self, x):
        for module in self:
            x = module(x)
        return x


class SequentialWithCondition(nn.Sequential):
    def forward(self, x, pixel_size):
        for module in self:
            x = module(x, pixel_size)
        return x


class Bottleneck(nn.Module):
    def __init__(
        self,
        in_channels,
        intermediate_channels,
        stride=1,
        expansion: int = 4,
        identity_downsample=None,
    ):
        super().__init__()
        self.expansion = expansion
        self.conv1 = nn.Conv2d(
            in_channels,
            intermediate_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )

        self.ln1 = ChannelwiseLayerNorm((intermediate_channels,))
        self.conv2 = nn.Conv2d(
            intermediate_channels,
            intermediate_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.ln2 = ChannelwiseLayerNorm((intermediate_channels,))
        self.conv3 = nn.Conv2d(
            intermediate_channels,
            intermediate_channels * self.expansion,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )

        self.ln3 = ChannelwiseLayerNorm((intermediate_channels * self.expansion,))
        self.relu = nn.LeakyReLU(0.2)
        self.ln4 = ChannelwiseLayerNorm((intermediate_channels * self.expansion,))
        self.identity_downsample = identity_downsample
        self.stride = stride

    def forward(self, x):
        identity = x.clone()

        x = self.conv1(x)
        x = self.ln1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.ln2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.ln3(x)

        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)
            identity = self.ln4(identity)

        x = x + identity
        x = self.relu(x)
        return x


class ConditionalBottleneck(nn.Module):
    """Bottleneck Block for ResNet50, ResNet101, ResNet152 with Conditional Normalization"""

    def __init__(
        self,
        in_channels,
        intermediate_channels,
        num_conditions,
        stride=1,
        expansion: int = 4,
        identity_downsample=None,
        variant: str = "conditional_normalization",
    ):
        super(ConditionalBottleneck, self).__init__()
        self.expansion = expansion
        self.identity_downsample = identity_downsample
        self.stride = stride
        self.variant = variant

        if variant == "conditional_normalization":
            # Use standard Conv with Conditional LayerNorm
            self.conv1 = nn.Conv2d(
                in_channels,
                intermediate_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            )
            self.conv2 = nn.Conv2d(
                intermediate_channels,
                intermediate_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
            )
            self.conv3 = nn.Conv2d(
                intermediate_channels,
                intermediate_channels * self.expansion,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            )
            num_of_features = 64 if intermediate_channels * expansion <= 256 else 128
            self.ln1 = ChannelwiseConditionalLayerNorm(
                (intermediate_channels,), num_conditions, num_of_features
            )
            self.ln2 = ChannelwiseConditionalLayerNorm(
                (intermediate_channels,), num_conditions, num_of_features
            )
            self.ln3 = ChannelwiseConditionalLayerNorm(
                (intermediate_channels * self.expansion,),
                num_conditions,
                num_of_features,
            )
            self.ln4 = ChannelwiseConditionalLayerNorm(
                (intermediate_channels * self.expansion,),
                num_conditions,
                num_of_features,
            )
        else:
            raise ValueError(f"Unknown variant: {variant}")

        self.relu = nn.LeakyReLU(0.2)

    def forward(self, x, pixel_size):
        identity = x.clone()

        if self.variant == "conditional_normalization":
            x = self.conv1(x)
            x = self.ln1(x, pixel_size)
            x = self.relu(x)
            x = self.conv2(x)
            x = self.ln2(x, pixel_size)
            x = self.relu(x)
            x = self.conv3(x)
            x = self.ln3(x, pixel_size)

            if self.identity_downsample is not None:
                identity = self.identity_downsample(identity)
                identity = self.ln4(identity, pixel_size)

        x = x + identity
        x = self.relu(x)
        return x
