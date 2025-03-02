import os
import torch
from torch.utils.data import Subset, Dataset
from torchvision.io import read_image
from torchvision import datasets, transforms
import random


classes = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)
class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}

CIFAR10Classes = classes
CIFAR10ClassToIdx = class_to_idx


class CIFAR10Dataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.samples = [
            s for s in os.listdir(root) if os.path.isfile(os.path.join(root, s))
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_name = self.samples[idx]
        label_name = img_name.split("_")[0]
        label = class_to_idx[label_name]
        img_path = os.path.join(self.root, img_name)
        image = read_image(img_path).float() / 255.0
        if self.transform:
            image = self.transform(image)
        return image, label


class CIFAR10Dataset_RandomShuffle(CIFAR10Dataset):
    def __init__(self, root, transform=None):
        super().__init__(root, transform)
        self.root = root
        self.transform = transform
        self.samples = []
        for split in ["train", "test"]:
            split_dir = os.path.join(root, split)
            if os.path.exists(split_dir):
                split_samples = [
                    os.path.join(split, s)
                    for s in os.listdir(split_dir)
                    if os.path.isfile(os.path.join(split_dir, s))
                ]
                self.samples.extend(split_samples)
        random.shuffle(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_name = self.samples[idx]
        label_name = img_name.split("/")[-1].split("_")[0]
        label = class_to_idx[label_name]
        img_path = os.path.join(self.root, img_name)
        image = read_image(img_path).float() / 255.0
        if self.transform:
            image = self.transform(image)
        return image, label

    def crop(self, size):
        self.samples = self.samples[:size]


# if __name__ == "__main__":
#     dataset = CIFAR10Dataset_RandomShuffle("CIFAR10")
#     print(len(dataset))

#     dataset = CIFAR10Dataset("CIFAR10/train")
#     print(len(dataset))

#     dataset = CIFAR10Dataset("CIFAR10/test")
#     print(len(dataset))
