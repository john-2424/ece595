import torch

from models.small_net import SmallNet

def shape(x): return list(x.shape)

m = SmallNet(num_classes=4)
total_params = sum(p.numel() for p in m.parameters())
trainable_params = sum(p.numel() for p in m.parameters() if p.requires_grad)

print(f"[Info] [Model] Total params: {total_params:,} | Trainable: {trainable_params:,}")
x = torch.randn(1,3,128,128)

# Stepwise trace
x = m.stem(x);   print(" stem   ->", shape(x))   # (1,32,128,128)
x = m.block1(x); print(" block1 ->", shape(x))  # (1,32, 64, 64)
x = m.block2(x); print(" block2 ->", shape(x))  # (1,64, 32, 32)
x = m.block3(x); print(" block3 ->", shape(x))  # (1,128,16, 16)
x = m.gap(x);    print(" gap    ->", shape(x))  # (1,128, 1, 1)
print("head -> [Flatten] -> [Dropout(0.1)] -> [Linear(128->4)]")
