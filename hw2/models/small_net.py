# models/small_net.py
import torch
import torch.nn as nn
from typing import Sequence

def conv_bn_relu(in_c: int, out_c: int, k: int = 3, s: int = 1, p: int = 1) -> nn.Sequential:
    """A tiny VGG-style unit: Conv -> BN -> ReLU."""
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size=k, stride=s, padding=p, bias=False),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
    )

class ConvBlock(nn.Module):
    """
    Two 3x3 Conv-BN-ReLU layers + (optional) residual if in_c==out_c,
    followed by 2x2 MaxPool to downsample.
    """
    def __init__(self, in_c: int, out_c: int, use_residual: bool = True):
        super().__init__()
        self.use_residual = use_residual and (in_c == out_c)
        self.body = nn.Sequential(
            conv_bn_relu(in_c, out_c, 3, 1, 1),
            conv_bn_relu(out_c, out_c, 3, 1, 1),
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.body(x)
        if self.use_residual:
            y = y + x
        y = self.pool(y)
        return y

class SmallNet(nn.Module):
    """
    Lightweight CNN for 4 classes.
    Input: 3x128x128 (we'll resize in the dataloader)
    Stages: [32] -> [64] -> [128], each a ConvBlock with MaxPool (total /8 downsample)
    Head: AdaptiveAvgPool(1) -> Dropout(0.1) -> Linear(128->4)
    """
    def __init__(self, num_classes: int = 4, in_channels: int = 3):
        super().__init__()
        chs: Sequence[int] = (32, 64, 128)

        self.stem = conv_bn_relu(in_channels, chs[0], 3, 1, 1)

        self.block1 = ConvBlock(chs[0], chs[0], use_residual=True)   # 32 -> 32
        self.block2 = ConvBlock(chs[0], chs[1], use_residual=False)  # 32 -> 64
        self.block3 = ConvBlock(chs[1], chs[2], use_residual=False)  # 64 -> 128

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.1),
            nn.Linear(chs[-1], num_classes),
        )

        # Kaiming init for convs, sensible init for BN/Linear
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)   # [B,32,H,W]
        x = self.block1(x) # [B,32,H/2,W/2]
        x = self.block2(x) # [B,64,H/4,W/4]
        x = self.block3(x) # [B,128,H/8,W/8]
        x = self.gap(x)    # [B,128,1,1]
        x = self.head(x)   # [B,4]
        return x
