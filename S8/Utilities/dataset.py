import torch
from torchvision import datasets


def get_dataloaders(
    train_transforms,
    test_transforms,
    shuffle=True,
    batch_size=64,
    num_workers=-1,
    pin_memory=True,
):
    train_data = datasets.CIFAR10(
        "../data", train=True, download=True, transform=train_transforms
    )
    test_data = datasets.CIFAR10(
        "../data", train=False, download=True, transform=test_transforms
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
