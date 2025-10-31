# train.py
import os
import time
import json
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from models.small_net import SmallNet
from utils.dataset import HW2Dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[Info] Using device: {device}")

def main():
    t_0 = time.time()

    # Dataset and Data Loader
    root = "data/h2-data"
    train_list = os.path.join(root, "train.txt")
    # test_list  = os.path.join(root, "test.txt")
    train_ds = HW2Dataset(root, train_list, train=True)
    # test_ds  = HW2Dataset(root, test_list, train=False)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=2)
    # test_loader  = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=2)

    # Model, loss, and optimizer
    model = SmallNet(num_classes=4).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Training loop
    num_epochs = 3
    for epoch in range(num_epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for imgs, labels in pbar:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * imgs.size(0)
            preds = outputs.argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            pbar.set_postfix(loss=loss.item(), acc=100*correct/total)
        print(f"[Info] [Train] [Epoch {epoch+1}] Loss={running_loss/total:.4f}, Acc={100*correct/total:.2f}%")

    # Evaluate
    """model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            preds = model(imgs).argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    acc = 100 * correct / total
    print(f"[Info] [Test] Acc={acc:.2f}%")"""

    # Save artifacts
    os.makedirs("artifacts", exist_ok=True)
    torch.save(model.state_dict(), "artifacts/model.pt")
    with open("artifacts/results.json", "w") as f:
        json.dump({
            "epochs": num_epochs,
            "train_time_s": time.time()-t_0,
            # "test_accuracy": acc
        }, f, indent=2)

if __name__ == "__main__":
    main()
