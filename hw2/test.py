# test.py
import torch
from models.small_net import SmallNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[info] Using device: {device}")

def main():
    model = SmallNet(num_classes=4).to(device)
    model.load_state_dict(torch.load("artifacts/model.pt", map_location="cpu"))
    model.eval()
    # TODO(step 5): real test loader on provided test list; compute accuracy
    print("TEST ACCURACY: 0.00%   (placeholder)")

if __name__ == "__main__":
    main()
