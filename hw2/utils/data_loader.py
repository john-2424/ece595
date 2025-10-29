# dataset_loader.py
import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

# Map folder names to numeric labels
CLASS_MAP = {
    "n02124075": 0,  # Egyptian cat
    "n07753592": 1,  # banana
    "n02504458": 2,  # African elephant
    "n03792782": 3,  # mountain bike
}

class ImageNet4Dataset(Dataset):
    def __init__(self, root_dir, list_file, train=True):
        """
        root_dir: path to 'h2-data'
        list_file: 'train.txt' or 'test.txt'
        train: bool → controls augmentation
        """
        self.root_dir = root_dir
        with open(list_file, "r") as f:
            self.files = [line.strip() for line in f if line.strip()]
        self.train = train

        # Normalization constants for ImageNet
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        if train:
            self.transform = transforms.Compose([
                transforms.Resize((144,144)),
                transforms.RandomResizedCrop(128, scale=(0.8,1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((128,128)),
                transforms.CenterCrop(128),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        # class id from filename prefix
        cls_token = fname.split("_")[0]
        label = CLASS_MAP.get(cls_token, -1)
        if label == -1:
            raise ValueError(f"Unknown class prefix in {fname}")

        img_path = os.path.join(self.root_dir, cls_token, fname)
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        return img, label
