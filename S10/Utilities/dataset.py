import numpy as np
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


def get_dataloaders(
    train_transforms,
    test_transforms,
    shuffle=True,
    batch_size=64,
    num_workers=-1,
    pin_memory=True,
):
    train_data = CIFAR10(
        datasets.CIFAR10("../data", train=True, download=True),
        transform=train_transforms,
    )
    test_data = CIFAR10(
        datasets.CIFAR10("../data", train=False, download=True),
        transform=test_transforms,
    )
    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, test_loader
