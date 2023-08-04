import numpy as np
import pytorch_lightning as pl
import torch
from torchvision import datasets


class CIFAR10(torch.utils.data.Dataset):
    def __init__(self, dataset, transform=None) -> None:
        # Initialize dataset and transform
        self.dataset = dataset
        self.transform = transform

    def __len__(self) -> int:
        # Return the length of the dataset
        return len(self.dataset)

    def __getitem__(self, index):
        # Get image and label
        image, label = self.dataset[index]

        # Convert PIL image to numpy array
        image = np.array(image)

        # Apply transformations
        if self.transform:
            image = self.transform(image=image)["image"]

        return (image, label)


class CIFAR10DataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_transforms,
        val_transforms,
        shuffle=True,
        data_dir="../data",
        batch_size=64,
        num_workers=-1,
        pin_memory=True,
    ):
        super().__init__()
        self.shuffle = shuffle
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.train_transforms = train_transforms
        self.val_transforms = val_transforms
        self.train_data = None
        self.val_data = None

    def prepare_data(self):
        datasets.CIFAR10(self.data_dir, train=True, download=True)
        datasets.CIFAR10(self.data_dir, train=False, download=True)

    def setup(self, stage):
        self.train_data = CIFAR10(
            datasets.CIFAR10(root=self.data_dir, train=True, download=False),
            transform=self.train_transforms,
        )
        self.val_data = CIFAR10(
            datasets.CIFAR10(root=self.data_dir, train=False, download=False),
            transform=self.val_transforms,
        )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_data,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_data,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
