# test.py
import torch
from models.small_net import SmallNet

def main():
    model = SmallNet(num_classes=4)
    model.load_state_dict(torch.load("artifacts/model.pt", map_location="cpu"))
    model.eval()
    # TODO(step 5): real test loader on provided test list; compute accuracy
    print("TEST ACCURACY: 0.00%   (placeholder)")

if __name__ == "__main__":
    main()
