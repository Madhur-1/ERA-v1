from typing import Any, Optional

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torchmetrics.functional import f1_score


class UNet(LightningModule):
    def __init__(
        self,
        max_filter_size,
        dropout=0,
        reduction_method="max_pool",
        expansion_method="upsample",
        loss_fn="cross_entropy",
        lr=1e-3,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.max_filter_size = max_filter_size
        self.lr = lr
        self.f1_score = f1_score
        if loss_fn == "cross_entropy":
            self.loss_fn = nn.CrossEntropyLoss()
        elif loss_fn == "dice":
            self.loss_fn = MulticlassDiceLoss(num_classes=3, softmax_dim=1)
        else:
            raise ValueError(f"Loss function {loss_fn} not supported")

        self.cblock1 = EncoderMiniBlock(
            3,
            max_filter_size // 16,
            dropout=dropout,
            channel_reduction_method=reduction_method,
        )
        self.cblock2 = EncoderMiniBlock(
            max_filter_size // 16,
            max_filter_size // 8,
            dropout=dropout,
            channel_reduction_method=reduction_method,
        )
        self.cblock3 = EncoderMiniBlock(
            max_filter_size // 8,
            max_filter_size // 4,
            dropout=dropout,
            channel_reduction_method=reduction_method,
        )
        self.cblock4 = EncoderMiniBlock(
            max_filter_size // 4,
            max_filter_size // 2,
            dropout=dropout,
            channel_reduction_method=reduction_method,
        )
        self.cblock5 = EncoderMiniBlock(
            max_filter_size // 2,
            max_filter_size // 2,
            dropout=dropout,
            channel_reduction_method=None,
        )

        self.ublock1 = DecoderMiniBlock(
            max_filter_size,
            max_filter_size // 2,
            dropout=dropout,
            channel_expansion_method=expansion_method,
        )
        self.ublock2 = DecoderMiniBlock(
            max_filter_size // 2,
            max_filter_size // 4,
            dropout=dropout,
            channel_expansion_method=expansion_method,
        )
        self.ublock3 = DecoderMiniBlock(
            max_filter_size // 4,
            max_filter_size // 8,
            dropout=dropout,
            channel_expansion_method=expansion_method,
        )
        self.ublock4 = DecoderMiniBlock(
            max_filter_size // 8,
            max_filter_size // 16,
            dropout=dropout,
            channel_expansion_method=expansion_method,
        )
        self.final_conv = nn.Conv2d(max_filter_size // 32, 3, 1)

    def forward(self, x):
        x, c1 = self.cblock1(x)
        x, c2 = self.cblock2(x)
        x, c3 = self.cblock3(x)
        x, c4 = self.cblock4(x)
        x, _ = self.cblock5(x)
        x = self.ublock1(x, c4)
        x = self.ublock2(x, c3)
        x = self.ublock3(x, c2)
        x = self.ublock4(x, c1)

        return self.final_conv(x)

    def configure_optimizers(self) -> Any:
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=2, verbose=True
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "train_loss_epoch",
        }

    def training_step(self, batch, batch_idx):
        inputs = batch["image"]
        labels = batch["mask"]
        labels = labels.squeeze(1)
        labels = labels.to(dtype=torch.long)

        outputs = self(inputs)
        loss = self.loss_fn(outputs, labels)
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        inputs = batch["image"]
        labels = batch["mask"]
        labels = labels.squeeze(1)
        labels = labels.to(dtype=torch.long)

        outputs = self(inputs)
        loss = self.loss_fn(outputs, labels)

        acc = (outputs.argmax(1) == labels).float().mean() * 100
        f1 = self.f1_score(outputs.argmax(1), labels, task="multiclass", num_classes=3)

        self.log(
            "val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        self.log(
            "val_acc", acc, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        self.log(
            "val_f1_score",
            f1,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return loss

    def on_train_epoch_end(self) -> None:
        print(
            f"Epoch: {self.current_epoch} Train Loss: {self.trainer.callback_metrics['train_loss_epoch']}"
        )

    def on_validation_epoch_end(self) -> None:
        print(
            f"Epoch: {self.current_epoch} Val Loss: {self.trainer.callback_metrics['val_loss_epoch']} Val Acc: {self.trainer.callback_metrics['val_acc_epoch']} Val F1: {self.trainer.callback_metrics['val_f1_score_epoch']}"
        )

    def plot_random_test_samples(self, num_samples=3):
        # Get num_samples random samples from the val dataset
        val_dataset = self.trainer.val_dataloaders.dataset
        val_samples = [
            val_dataset[i]
            for i in torch.randint(
                0,
                len(val_dataset),
                (num_samples,),
                generator=torch.Generator().manual_seed(42),
            )
        ]

        # Run the samples through the model and plot the results
        fig, axs = plt.subplots(num_samples, 3, figsize=(15, 15))
        for i, sample in enumerate(val_samples):
            image = sample["image"]
            mask = sample["mask"]

            mask = mask.squeeze(1)

            output = self(image.unsqueeze(0))
            output = output.argmax(1).squeeze(0)

            axs[i, 0].imshow(image.permute(1, 2, 0))
            axs[i, 1].imshow(mask.permute(1, 2, 0))
            axs[i, 2].imshow(output)

            axs[i, 0].set_title("Input Image")
            axs[i, 1].set_title("Ground Truth Mask")
            axs[i, 2].set_title("Predicted Mask")

            for ax in axs[i]:
                ax.set_xticks([])
                ax.set_yticks([])
        plt.tight_layout()
        plt.show()


class EncoderMiniBlock(LightningModule):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        channel_reduction_method: str = None,
        dropout: int = 0,
    ) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.PReLU()
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        if channel_reduction_method == "max_pool":
            self.channel_size_reduce = nn.MaxPool2d(kernel_size=2, stride=2)
        elif channel_reduction_method == "strided_conv":
            self.channel_size_reduce = nn.Conv2d(
                out_channels, out_channels, kernel_size=3, stride=2, padding=1
            )
        else:
            self.channel_size_reduce = nn.Identity()

    def forward(self, x):
        x = self.bn1(self.relu(self.conv1(x)))
        x = self.bn2(self.relu(self.conv2(x)))
        x = self.dropout(x)
        x1 = self.channel_size_reduce(x)

        # return x for skip connection
        return x1, x


class DecoderMiniBlock(LightningModule):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        channel_expansion_method: str = None,
        dropout: int = 0,
    ) -> None:
        super().__init__()
        if channel_expansion_method == "upsample":
            self.channel_size_expand = nn.Upsample(
                scale_factor=2, mode="bilinear", align_corners=True
            )
        elif channel_expansion_method == "strided_conv_transpose":
            self.channel_size_expand = nn.ConvTranspose2d(
                in_channels // 2,
                in_channels // 2,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
            )

        else:
            raise ValueError(
                f"Channel expansion method {channel_expansion_method} not supported"
            )

        self.dropout = nn.Dropout(dropout)
        self.relu = nn.PReLU()
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels // 2)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels // 2, kernel_size=3, padding=1
        )

    def forward(self, x, skip_layer_input):
        x = self.channel_size_expand(x)
        x = torch.cat([skip_layer_input, x], dim=1)
        x = self.bn1(self.relu(self.conv1(x)))
        x = self.bn2(self.relu(self.conv2(x)))
        x = self.dropout(x)
        return x


class MulticlassDiceLoss(nn.Module):
    """Reference: https://www.kaggle.com/code/bigironsphere/loss-function-library-keras-pytorch#Dice-Loss"""

    def __init__(self, num_classes, softmax_dim=None):
        super().__init__()
        self.num_classes = num_classes
        self.softmax_dim = softmax_dim

    def forward(self, logits, targets, reduction="mean", smooth=1e-6):
        """The "reduction" argument is ignored. This method computes the dice
        loss for all classes and provides an overall weighted loss.
        """
        probabilities = logits
        if self.softmax_dim is not None:
            probabilities = nn.Softmax(dim=self.softmax_dim)(logits)
        # end if
        targets_one_hot = torch.nn.functional.one_hot(
            targets, num_classes=self.num_classes
        )

        # Convert from NHWC to NCHW
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2)

        # Multiply one-hot encoded ground truth labels with the probabilities to get the
        # prredicted probability for the actual class.
        intersection = (targets_one_hot * probabilities).sum()

        mod_a = intersection.sum()
        mod_b = targets.numel()

        dice_coefficient = 2.0 * intersection / (mod_a + mod_b + smooth)
        dice_loss = -dice_coefficient.log()
        return dice_loss
