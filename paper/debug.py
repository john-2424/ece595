import torch
from models.convnext import convnext_tiny

model = convnext_tiny(num_classes=10)
x = torch.randn(2, 3, 32, 32)  # CIFAR-10 shape
y = model(x)
print(y.shape)   # should be [2, 10]
