import os
import random
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class OxfordIIITPetDataset(Dataset):
    def __init__(
        self, images: list[str], masks: list[str], transform=None, img_size=240
    ) -> None:
        super().__init__()
        self.images = images
        self.masks = masks
        self.transform = transform
        self.img_size = img_size

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image_path = self.images[index]
        mask_path = self.masks[index]

        image = Image.open(image_path)
        mask = Image.open(mask_path)

        if self.transform is None:
            self.transform = transforms.Compose(
                [
                    transforms.Resize((self.img_size, self.img_size)),
                    transforms.ToTensor(),
                ]
            )
        image = self.transform(image)
        mask = self.transform(mask) * 255.0 - 1.0

        return {"image": image, "mask": mask}

    def show_image_mask(self, index: int) -> None:
        image_path = self.images[index]
        mask_path = self.masks[index]

        image = Image.open(image_path).convert("RGB")
        mask = np.array(Image.open(mask_path)) * 80.0
        mask = Image.fromarray(mask).convert("RGB")
        mask.show()
        image.show()
        # mask.show()


class OxfordIIITPetDatamodule(LightningDataModule):
    def __init__(
        self,
        train_file: str,
        val_file: str,
        images_dir: str,
        masks_dir: str,
        batch_size: int,
        pin_memory: bool = True,
        random_seed: int = 42,
    ) -> None:
        super().__init__()
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.batch_size = batch_size
        self.pin_memory = pin_memory
        self.random_seed = random_seed

        self.train_file = train_file
        self.val_file = val_file
        self.train_images = []
        self.train_masks = []
        self.val_images = []
        self.val_masks = []

    def prepare_data(self) -> None:
        self.train_images = []
        self.train_masks = []
        self.val_images = []
        self.val_masks = []
        train_dataset = pd.read_csv(self.train_file, sep=" ", header=None)[0].to_list()
        val_dataset = pd.read_csv(self.val_file, sep=" ", header=None)[0].to_list()

        for train_file in train_dataset:
            self.train_images.append(os.path.join(self.images_dir, train_file + ".jpg"))
            self.train_masks.append(os.path.join(self.masks_dir, train_file + ".png"))

        for val_file in val_dataset:
            self.val_images.append(os.path.join(self.images_dir, val_file + ".jpg"))
            self.val_masks.append(os.path.join(self.masks_dir, val_file + ".png"))

        print("Training images:", len(self.train_images))
        print("Validation images:", len(self.val_images))

    def train_dataloader(self) -> DataLoader:
        train_dataset = OxfordIIITPetDataset(
            self.train_images,
            self.train_masks,
            transform=None,
            img_size=240,
        )
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            pin_memory=self.pin_memory,
            shuffle=True,
            num_workers=2,
        )
        return train_dataloader

    def val_dataloader(self) -> DataLoader:
        val_dataset = OxfordIIITPetDataset(
            self.val_images,
            self.val_masks,
            transform=None,
            img_size=240,
        )
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            pin_memory=self.pin_memory,
            shuffle=False,
            num_workers=2,
        )
        return val_dataloader
