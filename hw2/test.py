# test.py
import os
import json

import torch
from torch.utils.data import DataLoader

from models.small_net import SmallNet
from utils.dataset import HW2Dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[Info] Using device: {device}")

def main():
    root = "data/h2-data"
    test_list = os.path.join(root, "test.txt")
    test_ds = HW2Dataset(root, test_list, train=False)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=2)

    # model
    model = SmallNet(num_classes=4).to(device)
    state_path = "artifacts/model.pt"
    if not os.path.isfile(state_path):
        raise FileNotFoundError(
            f"Missing {state_path}. Run train.py (or ./run) to create it."
        )
    model.load_state_dict(torch.load(state_path, map_location=device))
    model.eval()

    # accuracy
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            preds = model(imgs).argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    acc = 100.0 * correct / total if total else 0.0
    print(f"[Info] [Test] Acc: {acc:.2f}%")

    # save a tiny report (optional but handy)
    os.makedirs("artifacts", exist_ok=True)
    with open("artifacts/test_report.json", "w") as f:
        json.dump({"test_accuracy": acc, "num_samples": total}, f, indent=2)

if __name__ == "__main__":
    main()
