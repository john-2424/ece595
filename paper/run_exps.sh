#!/bin/bash
set -e

# Common settings
DATA_DIR="./data"
EPOCHS=50
BATCH_SIZE=128

echo "Running experiments with data dir: ${DATA_DIR}"

# 1. ResNet-18 baseline
echo "=== ResNet-18 baseline ==="
python train_cifar10.py \
  --model resnet18 \
  --data-dir "${DATA_DIR}" \
  --epochs ${EPOCHS} \
  --batch-size ${BATCH_SIZE}

# 2. ConvNeXt-Tiny baseline (k7, LN, mlp4, patchify stem)
echo "=== ConvNeXt-Tiny baseline (k7, ln, mlp4, patchify) ==="
python train_cifar10.py \
  --model convnext_tiny \
  --data-dir "${DATA_DIR}" \
  --epochs ${EPOCHS} \
  --batch-size ${BATCH_SIZE} \
  --kernel-size 7 \
  --norm-layer ln \
  --mlp-ratio 4 \
  --stem-type patchify

# 3. Kernel size ablation: k3 vs k7
echo "=== Kernel ablation: k3, ln, mlp4, patchify ==="
python train_cifar10.py \
  --model convnext_tiny \
  --data-dir "${DATA_DIR}" \
  --epochs ${EPOCHS} \
  --batch-size ${BATCH_SIZE} \
  --kernel-size 3 \
  --norm-layer ln \
  --mlp-ratio 4 \
  --stem-type patchify

# (optional: add other kernel sizes if desired, e.g., k5, k9)

# 4. Norm ablation: LN vs BN
echo "=== Norm ablation: BN (k7, bn, mlp4, patchify) ==="
python train_cifar10.py \
  --model convnext_tiny \
  --data-dir "${DATA_DIR}" \
  --epochs ${EPOCHS} \
  --batch-size ${BATCH_SIZE} \
  --kernel-size 7 \
  --norm-layer bn \
  --mlp-ratio 4 \
  --stem-type patchify

# 5. MLP ratio ablation: mlp2, mlp4, mlp6
echo "=== MLP ablation: mlp2 (k7, ln, mlp2, patchify) ==="
python train_cifar10.py \
  --model convnext_tiny \
  --data-dir "${DATA_DIR}" \
  --epochs ${EPOCHS} \
  --batch-size ${BATCH_SIZE} \
  --kernel-size 7 \
  --norm-layer ln \
  --mlp-ratio 2 \
  --stem-type patchify

echo "=== MLP ablation: mlp6 (k7, ln, mlp6, patchify) ==="
python train_cifar10.py \
  --model convnext_tiny \
  --data-dir "${DATA_DIR}" \
  --epochs ${EPOCHS} \
  --batch-size ${BATCH_SIZE} \
  --kernel-size 7 \
  --norm-layer ln \
  --mlp-ratio 6 \
  --stem-type patchify

# 6. Stem type ablation: patchify vs resnet
echo "=== Stem ablation: ResNet-style stem (k7, ln, mlp4) ==="
python train_cifar10.py \
  --model convnext_tiny \
  --data-dir "${DATA_DIR}" \
  --epochs ${EPOCHS} \
  --batch-size ${BATCH_SIZE} \
  --kernel-size 7 \
  --norm-layer ln \
  --mlp-ratio 4 \
  --stem-type resnet

echo "All experiments finished."
