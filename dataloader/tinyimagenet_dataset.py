import io
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from .classes import names, i2d

label_ids = names


def get_label_names(label_ids):
    label_names = {}
    for label in label_ids:
        label_names[label] = i2d[label]
    return label_names


label_names = get_label_names(label_ids)


class ParquetDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        """
        Args:
            dataframe (pandas.DataFrame): Dataframe containing the image bytes and labels.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.df = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image_data = self.df.iloc[idx]
        image_bytes = image_data["image"][
            "bytes"
        ]  # Correctly access the bytes from the dictionary
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        label = image_data["label"]

        if self.transform:
            image = self.transform(image)

        return image, label


if __name__ == "__main__":
    # Load training data
    train_data = pd.read_parquet(
        "/Users/rayaq/Desktop/uWaterloo/FYDP/code/DistributedML/TinyImageNet/data/tiny-imagenet/data/train-00000-of-00001-1359597a978bc4fa.parquet"
    )
    # Load validation data
    valid_data = pd.read_parquet(
        "/Users/rayaq/Desktop/uWaterloo/FYDP/code/DistributedML/TinyImageNet/data/tiny-imagenet/data/valid-00000-of-00001-70d52db3c749a935.parquet"
    )

    # Define transformations
    transform = transforms.Compose(
        [
            transforms.Resize((64, 64)),  # Adjust the size as needed
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Create dataset instances
    train_dataset = ParquetDataset(train_data, transform=transform)
    valid_dataset = ParquetDataset(valid_data, transform=transform)

    # DataLoader setup
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=64, shuffle=False)

    # functions to show an image
    def imshow(img, label):
        # img = img * 0.5 + 0.5
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.axis("off")
        plt.title(f"Label: {label}")
        plt.show()

    # Example usage with the first batch from the DataLoader
    dataiter = iter(train_loader)
    images, labels = next(dataiter)

    # Display the first image in the batch
    l = label_names[label_ids[labels[0].item()]]
    imshow(images[0], l)
