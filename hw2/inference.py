# inference.py
import os
import random

from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torchvision import transforms

from models.small_net import SmallNet
from utils.dataset import CLASS_MAP

INV_CLASS_MAP = {v: k for k, v in CLASS_MAP.items()}
CLASS_NAMES = {
    0: "Egyptian cat",
    1: "Banana",
    2: "African elephant",
    3: "Mountain bike",
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[Info] Using device: {device}")

def load_model(weights_path="artifacts/model.pt"):
    model = SmallNet(num_classes=4)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device).eval()
    return model, device

def preprocess():
    return transforms.Compose([
        transforms.Resize((128,128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406],
                             std=[0.229,0.224,0.225]),
    ])

def predict_image(model, device, image_path, tfm):
    img = Image.open(image_path).convert("RGB")
    x = tfm(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1).squeeze(0).cpu()
        pred = int(probs.argmax().item())
        conf = float(probs[pred].item())
    return pred, conf, img

def sample_from_test(root_dir="data/h2-data", list_file="data/h2-data/test.txt", per_class=1):
    # Group paths by class from test.txt
    with open(list_file, "r") as f:
        files = [ln.strip() for ln in f if ln.strip()]
    by_cls = {k: [] for k in CLASS_MAP}
    for fname in files:
        cls = fname.split("_")[0]
        path = os.path.join(root_dir, cls, fname)
        if cls in by_cls:
            by_cls[cls].append(path)
    # Choose up to 'per_class' each
    samples = []
    for cls, paths in by_cls.items():
        if not paths: 
            continue
        picks = random.sample(paths, k=min(per_class, len(paths)))
        for p in picks:
            samples.append(p)
    return samples

def main():
    os.makedirs("artifacts", exist_ok=True)
    model, device = load_model()
    tfm = preprocess()
    samples = sample_from_test(per_class=1)
    if not samples:
        print("[Warn] No samples found for inference.")
        return

    cols = len(samples)
    plt.figure(figsize=(3.2*cols, 3.6))
    for i, path in enumerate(samples):
        cls_token = os.path.basename(path).split("_")[0]
        gt_idx = CLASS_MAP[cls_token]
        gt_name = CLASS_NAMES[gt_idx]

        pred_idx, conf, img = predict_image(model, device, path, tfm)
        pred_name = CLASS_NAMES[pred_idx]
        title = f"Ground Truth: {gt_name}\nPrediction: {pred_name} ({conf*100:.1f}%)"

        ax = plt.subplot(1, cols, i+1)
        ax.imshow(img)
        ax.set_title(title, fontsize=10)
        ax.axis("off")

    plt.tight_layout()
    out_path = "artifacts/inference_grid.png"
    plt.savefig(out_path, bbox_inches="tight", dpi=160)
    print(f"[Info] [Inference] Saved {out_path}")
    try:
        print("[Info] **** Close the Matplotlib figure window to continue! ****")
        plt.show()
    except Exception:
        pass

if __name__ == "__main__":
    main()
