# ----------------------------
#  ConvNeXt Model Definition
# ----------------------------
from typing import List

import torch
import torch.nn as nn

class LayerNorm2d(nn.Module):
    """
    LayerNorm over channel dimension for (B, C, H, W) tensors.
    Implemented via nn.LayerNorm on channels-last view.
    """
    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.ln = nn.LayerNorm(num_channels, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        b, c, h, w = x.shape
        x = x.permute(0, 2, 3, 1)  # (B, H, W, C)
        x = self.ln(x)
        x = x.permute(0, 3, 1, 2)  # back to (B, C, H, W)
        return x


class DropPath(nn.Module):
    """
    Stochastic depth per-sample (when applied in residual branch).
    From: https://arxiv.org/abs/1603.09382 (ResNet stochastic depth)
    """
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        # Work with (B, 1, 1, 1) broadcast
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        output = x / keep_prob * random_tensor
        return output


class ConvNeXtBlock(nn.Module):
    """
    ConvNeXt block:
    - LayerNorm2d
    - depthwise 7x7 conv
    - pointwise conv (C -> 4C)
    - GELU
    - pointwise conv (4C -> C)
    - residual + optional DropPath
    """
    def __init__(self, dim: int, drop_path: float = 0.0,
                 kernel_size: int = 7, mlp_ratio: int = 4):
        super().__init__()
        self.ln = LayerNorm2d(dim)
        self.dwconv = nn.Conv2d(
            dim, dim, kernel_size=kernel_size,
            padding=kernel_size // 2, groups=dim
        )  # depthwise
        self.pwconv1 = nn.Conv2d(dim, dim * mlp_ratio, kernel_size=1)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv2d(dim * mlp_ratio, dim, kernel_size=1)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.ln(x)
        x = self.dwconv(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = self.drop_path(x)
        return x + shortcut


class ConvNeXtStage(nn.Module):
    """
    One ConvNeXt stage:
    - (optional) downsampling (2x2 stride-2 conv) with LN
    - N ConvNeXt blocks at fixed channel width
    """
    def __init__(self, in_channels: int, out_channels: int,
                 depth: int, drop_path_rates: List[float],
                 downsample: bool = True):
        super().__init__()
        layers = []
        if downsample:
            layers.append(
                nn.Sequential(
                    LayerNorm2d(in_channels),
                    nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2)
                )
            )
            in_channels = out_channels
        else:
            assert in_channels == out_channels

        self.downsample = nn.Sequential(*layers) if layers else nn.Identity()

        blocks = []
        for i in range(depth):
            blocks.append(
                ConvNeXtBlock(out_channels, drop_path=drop_path_rates[i])
            )
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.downsample(x)
        x = self.blocks(x)
        return x


class ConvNeXt(nn.Module):
    """
    ConvNeXt backbone + classification head.
    This implementation roughly matches ConvNeXt-Tiny config:
    - depths: [3, 3, 9, 3]
    - dims:   [96, 192, 384, 768]
    """
    def __init__(
        self,
        in_chans: int = 3,
        num_classes: int = 10,
        depths: List[int] = [3, 3, 9, 3],
        dims: List[int] = [96, 192, 384, 768],
        drop_path_rate: float = 0.1
    ):
        super().__init__()
        assert len(depths) == len(dims) == 4

        self.num_classes = num_classes

        # Patchify stem: 4x4 conv, stride 4
        self.stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm2d(dims[0])
        )

        # stochastic depth schedule
        total_blocks = sum(depths)
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, total_blocks)]
        dp_index = 0

        stages = []
        in_dim = dims[0]
        for stage_idx in range(4):
            depth = depths[stage_idx]
            out_dim = dims[stage_idx]
            downsample = stage_idx > 0  # first stage already downsampled by stem

            stage_dp_rates = dp_rates[dp_index: dp_index + depth]
            dp_index += depth

            stages.append(
                ConvNeXtStage(
                    in_channels=in_dim,
                    out_channels=out_dim,
                    depth=depth,
                    drop_path_rates=stage_dp_rates,
                    downsample=downsample
                )
            )
            in_dim = out_dim

        self.stages = nn.Sequential(*stages)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.head_norm = nn.LayerNorm(dims[-1], eps=1e-6)
        self.head = nn.Linear(dims[-1], num_classes)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stages(x)
        x = self.global_pool(x)  # (B, C, 1, 1)
        x = x.flatten(1)         # (B, C)
        x = self.head_norm(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        x = self.head(x)
        return x


def convnext_tiny(num_classes: int = 10, in_chans: int = 3) -> ConvNeXt:
    """Factory function for ConvNeXt-Tiny-like model."""
    return ConvNeXt(
        in_chans=in_chans,
        num_classes=num_classes,
        depths=[3, 3, 9, 3],
        dims=[96, 192, 384, 768],
        drop_path_rate=0.1
    )
