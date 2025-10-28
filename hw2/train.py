# train.py
import json, os, time
import argparse

import torch

from models.small_net import SmallNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[info] Using device: {device}")

def main():
    t0 = time.time()
    model = SmallNet(num_classes=4).to(device)
    # TODO(step 3): build real Dataset/DataLoader for train split
    # TODO(step 4): training loop for 2-3 quick epochs (CPU-friendly)
    # Save minimal artifacts the grader can use:
    os.makedirs("artifacts", exist_ok=True)
    torch.save(model.state_dict(), "artifacts/model.pt")
    with open("artifacts/results.json", "w") as f:
        json.dump({"note": "placeholder; training to be added", "train_time_s": time.time()-t0}, f)
    print("[train] done")

if __name__ == "__main__":
    main()
