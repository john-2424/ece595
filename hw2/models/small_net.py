# models/small_net.py
import torch
import torch.nn as nn

class SmallNet(nn.Module):
    def __init__(self, num_classes: int = 4):
        super().__init__()
        # TODO(step 2): implement three conv blocks with 3x3 conv + BN + ReLU + MaxPool
        # Keep channels modest (e.g., 32 -> 64 -> 128), then GlobalAvgPool -> Linear(num_classes)
        # VGG-style stacks of small 3x3s are parameter-efficient and effective【:contentReference[oaicite:2]{index=2}】.
        self.net = nn.Identity()
        self.head = nn.Linear(1, num_classes)  # placeholder

    def forward(self, x):
        # TODO(step 2): replace with real forward pass
        x = torch.mean(x, dim=(2,3), keepdim=False)  # bogus GAP over raw input (placeholder)
        return self.head(x)
