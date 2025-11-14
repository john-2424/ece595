# ---------------------------------------
#  Minimal Training Script (CIFAR-10)
# ---------------------------------------
import os
import csv
import argparse

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import resnet18

from models.convnext import convnext_tiny


def get_cifar10_loaders(data_dir: str, batch_size: int, num_workers: int = 4):
    """Create train/val loaders for CIFAR-10."""
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2470, 0.2435, 0.2616),
        ),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2470, 0.2435, 0.2616),
        ),
    ])

    train_dataset = datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=transform_train
    )
    test_dataset = datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=transform_test
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    return train_loader, test_loader


def train_one_epoch(model, optimizer, dataloader, device, epoch, scaler=None):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, targets in dataloader:
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        if scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = F.cross_entropy(outputs, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = F.cross_entropy(outputs, targets)
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total * 100.0
    print(f"Epoch {epoch} | Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.2f}%")
    return epoch_loss, epoch_acc


def evaluate(model, dataloader, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            targets = targets.to(device)

            outputs = model(images)
            loss = F.cross_entropy(outputs, targets)
            running_loss += loss.item() * images.size(0)

            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    loss = running_loss / total
    acc = correct / total * 100.0
    return loss, acc


def main():
    parser = argparse.ArgumentParser(description="ConvNeXt-Tiny on CIFAR-10")
    parser.add_argument("--model", type=str, default="convnext_tiny",
                        choices=["convnext_tiny", "resnet18"])
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--no-amp", action="store_true", help="Disable mixed precision")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    train_loader, test_loader = get_cifar10_loaders(
        args.data_dir, args.batch_size, num_workers=args.num_workers
    )

    # Choose model
    if args.model == "resnet18":
        model = resnet18(weights=None, num_classes=10)
    elif args.model == "convnext_tiny":
        model = convnext_tiny(num_classes=10, in_chans=3)

    model = model.to(device)
    
    os.makedirs("results", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    log_path = os.path.join("results", f"{args.model}_cifar10_log.csv")
    ckpt_path = os.path.join("checkpoints", f"{args.model}_cifar10_best.pth")

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {args.model} | Trainable parameters: {num_params/1e6:.2f}M")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )

    scaler = None
    if (device == "cuda") and (not args.no_amp):
        scaler = torch.cuda.amp.GradScaler()

    best_acc = 0.0
    
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, optimizer, train_loader, device, epoch, scaler=scaler
        )
        val_loss, val_acc = evaluate(model, test_loader, device)
        scheduler.step()

        print(
            f"Epoch {epoch} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%"
        )

        with open(log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, train_loss, train_acc, val_loss, val_acc])

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "epoch": epoch,
                    "val_acc": val_acc,
                },
                ckpt_path,
            )
            print(
                f"New best model saved with val acc: {best_acc:.2f}% "
                f"-> {ckpt_path}"
            )

    print(f"Training complete. Best Val Acc: {best_acc:.2f}%")
    print(f"Full training log saved to: {log_path}")
    print(f"Best checkpoint saved to: {ckpt_path}")


if __name__ == "__main__":
    main()
