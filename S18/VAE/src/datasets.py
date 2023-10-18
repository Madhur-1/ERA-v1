import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

# Note - you must have torchvision installed for this example
from torchvision.datasets import MNIST


class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = "../../data/", batch_size: int = 32) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )
        self.batch_size = batch_size

    def prepare_data(self):
        # download
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            mnist_full = MNIST(self.data_dir, train=True, transform=self.transform)
            self.mnist_train, self.mnist_val = random_split(
                mnist_full, [55000, 5000], generator=torch.Generator().manual_seed(42)
            )

        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            self.mnist_test = MNIST(
                self.data_dir, train=False, transform=self.transform
            )

        if stage == "predict":
            self.mnist_predict = MNIST(
                self.data_dir, train=False, transform=self.transform
            )

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=self.batch_size)

    def predict_dataloader(self):
        return DataLoader(self.mnist_predict, batch_size=self.batch_size)


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
        data_dir="../../data",
        batch_size=64,
        num_workers=4,
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
