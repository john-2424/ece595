# train.py
import os, json, time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.small_net import SmallNet
from utils.data_loader import ImageNet4Dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[info] Using device: {device}")

def main():
    t0 = time.time()

    # 1) Datasets and loaders
    root = "data/h2-data"
    train_list = os.path.join(root, "train.txt")
    test_list  = os.path.join(root, "test.txt")

    train_ds = ImageNet4Dataset(root, train_list, train=True)
    test_ds  = ImageNet4Dataset(root, test_list, train=False)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=2)
    test_loader  = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=2)

    # 2) Model, loss, optimizer
    model = SmallNet(num_classes=4).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # 3) Training loop (2-3 epochs only)
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
        print(f"[epoch {epoch+1}] loss={running_loss/total:.4f}, acc={100*correct/total:.2f}%")

    # 4) Evaluate quickly
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            preds = model(imgs).argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    acc = 100 * correct / total
    print(f"[test] accuracy={acc:.2f}%")

    # 5) Save artifacts
    os.makedirs("artifacts", exist_ok=True)
    torch.save(model.state_dict(), "artifacts/model.pt")
    with open("artifacts/results.json", "w") as f:
        json.dump({
            "epochs": num_epochs,
            "train_time_s": time.time()-t0,
            "test_accuracy": acc
        }, f, indent=2)

if __name__ == "__main__":
    main()
