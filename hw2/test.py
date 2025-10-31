# test.py
import os
import json
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
plt.ion()

import torch
from torch.utils.data import DataLoader

from models.small_net import SmallNet
from utils.dataset import HW2Dataset

CLASS_NAMES = {
    0: "Egyptian Cat",
    1: "Banana",
    2: "African Elephant",
    3: "Mountain Bike",
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[Info] Using device: {device}")

def metrics_from_confusion(cm: np.ndarray):
    """Compute per-class TP/FP/FN/TN, precision/recall/F1, plus macro/micro F1."""
    K = cm.shape[0]
    per_class = []
    total = cm.sum()
    for i in range(K):
        TP = int(cm[i, i])
        FP = int(cm[:, i].sum() - TP)
        FN = int(cm[i, :].sum() - TP)
        TN = int(total - TP - FP - FN)

        prec = TP / (TP + FP) if (TP + FP) else 0.0
        rec  = TP / (TP + FN) if (TP + FN) else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        acc_i = (TP + TN) / total if total else 0.0

        per_class.append({
            "TP": TP, "FP": FP, "FN": FN, "TN": TN,
            "precision": prec, "recall": rec, "f1": f1, "class_accuracy": acc_i
        })

    macro_precision = float(np.mean([c["precision"] for c in per_class]))
    macro_recall    = float(np.mean([c["recall"] for c in per_class]))
    macro_f1        = float(np.mean([c["f1"] for c in per_class]))

    TP_sum = float(np.trace(cm))
    FP_sum = float(cm.sum(axis=0).sum() - np.trace(cm))
    FN_sum = float(cm.sum(axis=1).sum() - np.trace(cm))

    micro_precision = TP_sum / (TP_sum + FP_sum) if (TP_sum + FP_sum) else 0.0
    micro_recall    = TP_sum / (TP_sum + FN_sum) if (TP_sum + FN_sum) else 0.0
    micro_f1        = 2 * micro_precision * micro_recall / (micro_precision + micro_recall) if (micro_precision + micro_recall) else 0.0

    return {
        "per_class": per_class,
        "macro": {"precision": macro_precision, "recall": macro_recall, "f1": macro_f1},
        "micro": {"precision": micro_precision, "recall": micro_recall, "f1": micro_f1},
    }

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

    # Collect predictions and labels; compute top-1 and top-2 accuracy
    all_preds, all_labels = [], []
    top1_correct, top2_correct, total = 0, 0, 0

    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs)

            preds = logits.argmax(1)
            _, topk = logits.topk(k=2, dim=1)  # for top-2 accuracy

            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

            top1_correct += (preds == labels).sum().item()
            top2_correct += (topk.eq(labels.unsqueeze(1))).any(dim=1).sum().item()
            total += labels.size(0)

    all_preds = np.concatenate(all_preds) if all_preds else np.array([], dtype=int)
    all_labels = np.concatenate(all_labels) if all_labels else np.array([], dtype=int)

    # Confusion matrix (true rows, predicted cols)
    K = 4
    cm = np.zeros((K, K), dtype=int)
    for y, yhat in zip(all_labels, all_preds):
        cm[y, yhat] += 1

    acc_top1 = 100.0 * top1_correct / total if total else 0.0
    acc_top2 = 100.0 * top2_correct / total if total else 0.0
    m = metrics_from_confusion(cm)

    # Console summary
    print(f"[Info] [Test] Acc@1: {acc_top1:.2f}% | Acc@2: {acc_top2:.2f}%")
    for i in range(K):
        name = CLASS_NAMES[i]
        prec = m["per_class"][i]["precision"]
        rec  = m["per_class"][i]["recall"]
        f1   = m["per_class"][i]["f1"]
        print(f"  - {name:17s}  P={prec:.3f}  R={rec:.3f}  F1={f1:.3f}")

    # Save artifacts
    os.makedirs("artifacts", exist_ok=True)
    np.savetxt("artifacts/confusion_matrix.csv", cm, fmt="%d", delimiter=",")
    with open("artifacts/metrics.json", "w") as f:
        json.dump({
            "accuracy_top1": acc_top1,
            "accuracy_top2": acc_top2,
            "confusion_matrix": cm.tolist(),
            **m
        }, f, indent=2)
    
    try:
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(K)); ax.set_yticks(range(K))
        ax.set_xticklabels([CLASS_NAMES[i] for i in range(K)], rotation=45, ha="right")
        ax.set_yticklabels([CLASS_NAMES[i] for i in range(K)])
        for i in range(K):
            for j in range(K):
                ax.text(j, i, cm[i, j], ha="center", va="center", color="black")
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig("artifacts/confusion_matrix.png", dpi=160, bbox_inches="tight")
        try:
            plt.show(block=False)
        except Exception:
            plt.close(fig)
    except Exception as e:
        print(f"[Warn] Could not save confusion_matrix.png ({e}). Skipping heatmap.")

    # Keep your original brief JSON report too
    with open("artifacts/test_report.json", "w") as f:
        json.dump({"test_accuracy": acc_top1, "num_samples": total}, f, indent=2)

if __name__ == "__main__":
    main()
