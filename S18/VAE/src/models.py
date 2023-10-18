import copy
import random

import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch

# from pl_bolts.callbacks.ssl_online import SSLOnlineEvaluator
# from pl_bolts.models.autoencoders.components import resnet18_decoder, resnet18_encoder
from torch import nn
from torch.nn import functional as F
from torch.optim.lr_scheduler import OneCycleLR
from torchvision import transforms

CLASSES = (
    "Airplane",
    "Automobile",
    "Bird",
    "Cat",
    "Deer",
    "Dog",
    "Frog",
    "Horse",
    "Ship",
    "Truck",
)
cifar_normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
cifar_inverse_normalize = transforms.Normalize(
    (-0.4914 / 0.247, -0.4822 / 0.243, -0.4465 / 0.261),
    (1 / 0.247, 1 / 0.243, 1 / 0.261),
)


class MNIST_Encoder(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm2d(10),
            nn.Conv2d(10, 10, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm2d(10),
            nn.Conv2d(10, 11, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm2d(11),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(11, 10, kernel_size=1),
            # nn.ReLU(),
            # nn.BatchNorm2d(10),
            nn.Conv2d(10, 10, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm2d(10),
            nn.Conv2d(10, 10, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm2d(10),
            nn.Conv2d(10, 11, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm2d(11),
            nn.Conv2d(11, 10, kernel_size=1),
        )
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pool2d(x, 2)
        x = self.conv2(x)
        x = self.avgpool(x)
        x = x.view(-1, 10)
        return x


class MNIST_Decoder(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=10, out_channels=11, kernel_size=3
            ),  # Input size = 1x1x10 -> 3x3x11
            nn.ReLU(),
            nn.BatchNorm2d(11),
            nn.ConvTranspose2d(
                in_channels=11, out_channels=10, kernel_size=3
            ),  # Input size = 3x3x11 -> 5x5x10
            nn.ReLU(),
            nn.BatchNorm2d(10),
            nn.ConvTranspose2d(
                in_channels=10, out_channels=10, kernel_size=3
            ),  # Input size = 5x5x10 -> 7x7x10
            nn.ReLU(),
            nn.BatchNorm2d(10),
            nn.ConvTranspose2d(
                in_channels=10, out_channels=11, kernel_size=3
            ),  # Input size = 7x7x10 -> 9x9x11
            nn.ReLU(),
            nn.BatchNorm2d(11),
            nn.ConvTranspose2d(
                in_channels=11, out_channels=11, kernel_size=3
            ),  # Input size = 9x9x11 -> 11x11x11
        )
        self.conv2 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=11, out_channels=10, kernel_size=3
            ),  # Input size = 22x22x11 -> 24x24x10
            nn.ReLU(),
            nn.BatchNorm2d(10),
            nn.ConvTranspose2d(
                in_channels=10, out_channels=10, kernel_size=3
            ),  # Input size = 24x24x10 -> 26x26x10
            nn.ReLU(),
            nn.BatchNorm2d(10),
            nn.ConvTranspose2d(
                in_channels=10, out_channels=1, kernel_size=3
            ),  # Input size = 26x26x10 -> 28x28x1
        )
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

    def forward(self, x):
        x = x.view(-1, 10, 1, 1)  # Input size = 10 -> 1x1x10
        x = self.conv1(x)  # Input size = 1x1x10 -> 11x11x11
        x = self.upsample(x)  # Input size = 11x11x11 -> 22x22x11
        x = self.conv2(x)  # Input size = 22x22x11 -> 28x28x1
        return x


class CIFAR_Encoder(pl.LightningModule):
    def __init__(self, latent_dim, dropout_percentage=0.1, padding=1):
        super().__init__()

        # Prep Layer
        self.prep_layer = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),  # 32x32x3 | 1 -> 32x32x64 | 3
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(dropout_percentage),
        )

        self.l1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # 32x32x128 | 5
            nn.MaxPool2d(2, 2),  # 16x16x128 | 6
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(dropout_percentage),
        )
        self.l1res = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),  # 16x16x128 | 10
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(dropout_percentage),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),  # 16x16x128 | 14
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(dropout_percentage),
        )
        self.l2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),  # 16x16x256 | 18
            nn.MaxPool2d(2, 2),  # 8x8x256 | 19
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(dropout_percentage),
        )
        self.l3 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),  # 8x8x512 | 27
            nn.MaxPool2d(2, 2),  # 4x4x512 | 28
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Dropout(dropout_percentage),
        )
        self.l3res = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),  # 4x4x512 | 36
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Dropout(dropout_percentage),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),  # 4x4x512 | 44
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Dropout(dropout_percentage),
        )
        self.maxpool = nn.MaxPool2d(4, 4)

        # Classifier
        self.linear = nn.Linear(512, latent_dim)

    def forward(self, x):
        x = self.prep_layer(x)
        x = self.l1(x)
        x = x + self.l1res(x)
        x = self.l2(x)
        x = self.l3(x)
        x = x + self.l3res(x)
        x = self.maxpool(x)
        x = x.view(-1, 512)
        x = self.linear(x)
        return x


class CIFAR_Decoder(pl.LightningModule):
    def __init__(self, latent_dim, dropout_percentage=0.1, padding=1):
        super().__init__()
        self.dropout_percentage = dropout_percentage
        self.padding = padding

        self.linear = nn.Linear(latent_dim, 512)

        self.upscale = nn.ConvTranspose2d(
            512, 512, kernel_size=4, stride=1
        )  # 1x1x512 -> 4x4x512

        self.l3res = nn.Sequential(
            nn.ConvTranspose2d(
                512, 512, kernel_size=3, padding=1
            ),  # 4x4x512 -> 4x4x512
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Dropout(dropout_percentage),
            nn.ConvTranspose2d(
                512, 512, kernel_size=3, padding=1
            ),  # 4x4x512 -> 4x4x512
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Dropout(dropout_percentage),
        )

        self.l3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),  # 4x4x512 -> 8x8x512
            nn.ConvTranspose2d(
                512, 256, kernel_size=3, padding=1
            ),  # 8x8x512 -> 8x8x256
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(dropout_percentage),
        )

        self.l2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),  # 8x8x256 -> 16x16x256
            nn.ConvTranspose2d(
                256, 128, kernel_size=3, padding=1
            ),  # 16x16x256 -> 16x16x128
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(dropout_percentage),
        )

        self.l1res = nn.Sequential(
            nn.ConvTranspose2d(
                128, 128, kernel_size=3, padding=1
            ),  # 16x16x128 -> 16x16x128
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(dropout_percentage),
            nn.ConvTranspose2d(
                128, 128, kernel_size=3, padding=1
            ),  # 16x16x128 -> 16x16x128
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(dropout_percentage),
        )

        self.l1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),  # 16x16x128 -> 32x32x128
            nn.ConvTranspose2d(
                128, 64, kernel_size=3, padding=1
            ),  # 32x32x128 -> 32x32x64
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(dropout_percentage),
        )

        self.prep_layer = nn.Sequential(
            nn.ConvTranspose2d(64, 3, kernel_size=3, padding=1),  # 32x32x64 -> 32x32x3
        )

    def forward(self, x):
        x = self.linear(x)
        x = x.view(-1, 512, 1, 1)
        x = self.upscale(x)  # 1x1x512 -> 4x4x512
        x = x + self.l3res(x)  # 4x4x512 -> 4x4x512
        x = self.l3(x)  # 4x4x512 -> 8x8x256
        x = self.l2(x)  # 8x8x256 -> 16x16x128
        x = x + self.l1res(x)  # 16x16x128 -> 16x16x128
        x = self.l1(x)  # 16x16x128 -> 32x32x64
        x = self.prep_layer(x)  # 32x32x64 -> 32x32x3
        # x = F.sigmoid(x)
        # Get the image back to normalized range
        # x = transforms.Normalize(0.5, 0.5)(x)
        return x


class VAE(pl.LightningModule):
    def __init__(
        self,
        encoder,
        decoder,
        enc_out_dim=10,
        latent_dim=5,
        lamda_label_loss=10,
    ):
        super().__init__()

        # self.save_hyperparameters()

        # encoder, decoder
        self.encoder = encoder
        self.decoder = decoder

        # distribution parameters
        self.fc_mu = nn.Linear(enc_out_dim, latent_dim)
        self.fc_var = nn.Linear(enc_out_dim, latent_dim)

        # for the gaussian likelihood
        self.log_scale = nn.Parameter(torch.Tensor([0.0]))

        # for label_loss
        # make a copy of the encoder

        self.decoder_classifier = copy.deepcopy(encoder)
        # set the grad to false
        self.decoder_classifier.requires_grad_(False)
        self.decoder_classifier_loss = F.nll_loss
        self.lamda_label_loss = lamda_label_loss

        # y input
        self.y_input_embed_mu = nn.Embedding(10, latent_dim)
        self.y_input_embed_log_var = nn.Embedding(10, latent_dim)

        # LR
        self.lr = 1e-4
        self.best_lr = 4e-4

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = OneCycleLR(
            optimizer=optimizer,
            max_lr=self.best_lr,
            steps_per_epoch=len(self.trainer.datamodule.train_dataloader()),
            epochs=self.trainer.max_epochs,
            pct_start=0.2,
            div_factor=100,
            final_div_factor=100,
            three_phase=False,
            anneal_strategy="linear",
        )
        return [optimizer], [
            {"scheduler": scheduler, "interval": "step", "frequency": 1}
        ]

    def gaussian_likelihood(self, mean, logscale, sample):
        scale = torch.exp(logscale)
        dist = torch.distributions.Normal(mean, scale)
        log_pxz = dist.log_prob(sample)
        return log_pxz.sum(dim=(1, 2, 3))

    def kl_divergence(self, z, mu, std):
        # --------------------------
        # Monte carlo KL divergence
        # --------------------------
        # 1. define the first two probabilities (in this case Normal for both)
        p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        q = torch.distributions.Normal(mu, std)

        # 2. get the probabilities from the equation
        log_qzx = q.log_prob(z)
        log_pz = p.log_prob(z)

        # kl
        kl = log_qzx - log_pz
        kl = kl.sum(-1)
        return kl

    def common_step(self, batch, batch_idx, random_label_prob=0.1):
        x, y = batch

        # for 10% of the batch, sample a random label
        y_input = y.clone()
        idx = random.sample(range(0, len(y)), int(random_label_prob * len(y)))
        y_input[idx] = torch.randint(0, 10, (len(idx),)).to(self.device)

        mu, log_var = self.y_input_embed_mu(y_input), self.y_input_embed_log_var(
            y_input
        )

        # encode x to get the mu and variance parameters
        x_encoded = self.encoder(x)
        # print("encoded shape", x_encoded.shape)
        mu += self.fc_mu(x_encoded)
        log_var += self.fc_var(x_encoded)

        # sample z from q
        std = torch.exp(log_var / 2)
        q = torch.distributions.Normal(mu, std)
        z = q.rsample()

        # decoded
        x_hat = self.decoder(z)
        return x_hat, x, z, mu, std, y, y_input

    def training_step(self, batch, batch_idx):
        x_hat, x, z, mu, std, y, y_input = self.common_step(batch, batch_idx)
        # print(x_hat.shape, x.shape)
        # reconstruction loss
        recon_loss = self.gaussian_likelihood(x_hat, self.log_scale, x)

        # kl
        kl = self.kl_divergence(z, mu, std)

        # elbo
        elbo = kl - recon_loss
        elbo = elbo.mean()

        # label classification loss
        y_hat = F.log_softmax(self.decoder_classifier(x_hat), dim=1)
        label_loss = self.decoder_classifier_loss(y_hat, y)
        scaled_label_loss = label_loss * self.lamda_label_loss
        total_loss = elbo + scaled_label_loss

        self.log_dict(
            {
                "loss": total_loss,
                "elbo": elbo,
                "kl": kl.mean(),
                "recon_loss": recon_loss.mean(),
                "scaled_label_loss": scaled_label_loss,
                # "reconstruction": recon_loss.mean(),
                # "kl": kl.mean(),
            },
            prog_bar=True,
        )

        return total_loss

    def plot_singe_test_mnist(self, sample_num, y_input):
        val_loader = self.trainer.datamodule.val_dataloader()
        x_test, y_test = next(iter(val_loader))
        # get the nth sample
        x_test, y_test = x_test[sample_num], y_test[sample_num]

        x_test, y_test = x_test.to(self.device), y_test.to(self.device)

        self.eval()
        y_input = torch.tensor(y_input).unsqueeze(0).to(self.device)
        mu, log_var = self.y_input_embed_log_var(y_input), self.y_input_embed_log_var(
            y_input
        )

        # encode x to get the mu and variance parameters
        x_encoded = self.encoder(x_test.unsqueeze(0))
        # print("encoded shape", x_encoded.shape)
        mu += self.fc_mu(x_encoded)
        log_var += self.fc_var(x_encoded)

        # sample z from q
        std = torch.exp(log_var / 2)
        q = torch.distributions.Normal(mu, std)
        z = q.rsample()

        # decoded
        x_hat = self.decoder(z)

        # transpose the dimensions to be able to plot them

        # combine the two images in one plot side by side
        fig, (ax1, ax2) = plt.subplots(1, 2)
        ax1.imshow(
            x_test.cpu().squeeze(),
            cmap="gray",
        )
        ax1.set_title("Original Image, original label: " + str(y_test.item()))

        ax2.imshow(
            x_hat.squeeze().cpu().detach().numpy(),
            cmap="gray",
        )
        ax2.set_title("Reconstructed Image, input label: " + str(y_input.item()))
        plt.tight_layout()
        plt.show()

    def plot_single_test_cifar(self, sample_num, y_input):
        val_loader = self.trainer.datamodule.val_dataloader()
        x_test, y_test = next(iter(val_loader))
        # get the nth sample
        x_test, y_test = x_test[sample_num], y_test[sample_num]
        # print(x_test[0])

        import matplotlib.pyplot as plt

        x_test, y_test = x_test.to(self.device), y_test.to(self.device)

        self.eval()
        y_input = torch.tensor(y_input).unsqueeze(0).to(self.device)
        mu, log_var = self.y_input_embed_log_var(y_input), self.y_input_embed_log_var(
            y_input
        )

        # encode x to get the mu and variance parameters
        x_encoded = self.encoder(x_test.unsqueeze(0))
        # print("encoded shape", x_encoded.shape)
        mu += self.fc_mu(x_encoded)
        log_var += self.fc_var(x_encoded)

        # sample z from q
        std = torch.exp(log_var / 2)
        q = torch.distributions.Normal(mu, std)
        z = q.rsample()

        # decoded
        x_hat = self.decoder(z)

        x_test = cifar_inverse_normalize(x_test)
        x_hat = cifar_inverse_normalize(x_hat)

        # combine the two images in one plot side by side
        fig, (ax1, ax2) = plt.subplots(1, 2)
        ax1.imshow(
            x_test.permute(1, 2, 0).cpu().squeeze(),
            # cmap="gray",
        )
        ax1.set_title("Orig Img, original label: " + str(CLASSES[y_test.item()]))

        ax2.imshow(
            x_hat.squeeze().permute(1, 2, 0).cpu().detach().numpy(),
            # cmap="gray",
        )
        ax2.set_title("Recon Img, input label: " + str(CLASSES[y_input.item()]))
        plt.tight_layout()
        plt.show()

    def plot_test_mnist(self, num_imgs_to_plot: int = 10, random_label_prob: int = 0.5):
        x_test, y_test = next(iter(self.trainer.datamodule.val_dataloader()))

        # select random images
        idx = random.sample(range(0, len(x_test)), num_imgs_to_plot)
        x_test, y_test = x_test[idx], y_test[idx]

        x_test, y_test = x_test.to(self.device), y_test.to(self.device)

        self.eval()
        x_hat, x, z, mu, std, y, y_input = self.common_step(
            (x_test, y_test), 0, random_label_prob=random_label_prob
        )

        import matplotlib.pyplot as plt

        # Plot side by side
        fig, axs = plt.subplots(num_imgs_to_plot, 2, figsize=(10, 20))
        for i in range(num_imgs_to_plot):
            axs[i, 0].imshow(
                x_test[i].cpu().squeeze(),
                cmap="gray",
            )
            axs[i, 0].set_title("Orig Img, original label: " + str(y_test[i].item()))
            axs[i, 1].imshow(
                x_hat[i].cpu().squeeze().detach().numpy(),
                cmap="gray",
            )
            axs[i, 1].set_title("Recon Img, input label: " + str(y_input[i].item()))
        plt.tight_layout()
        plt.show()

    def plot_test_cifar(self, num_imgs_to_plot: int = 10, random_label_prob: int = 0.5):
        x_test, y_test = next(iter(self.trainer.datamodule.val_dataloader()))

        # select random images
        idx = random.sample(range(0, len(x_test)), num_imgs_to_plot)
        x_test, y_test = x_test[idx], y_test[idx]

        x_test, y_test = x_test.to(self.device), y_test.to(self.device)

        self.eval()
        x_hat, x, z, mu, std, y, y_input = self.common_step(
            (x_test, y_test), 0, random_label_prob=random_label_prob
        )

        x_test = cifar_inverse_normalize(x_test)
        x_hat = cifar_inverse_normalize(x_hat)
        # Plot side by side
        fig, axs = plt.subplots(num_imgs_to_plot, 2, figsize=(10, 20))
        for i in range(num_imgs_to_plot):
            axs[i, 0].imshow(
                x_test[i].permute(1, 2, 0).cpu().squeeze(),
                # cmap="gray",
            )
            axs[i, 0].set_title(
                "Orig Img, original label: " + str(CLASSES[y_test[i].item()])
            )
            axs[i, 1].imshow(
                x_hat[i].permute(1, 2, 0).cpu().squeeze().detach().numpy(),
                # cmap="gray",
            )
            axs[i, 1].set_title(
                "Recon Img, input label: " + str(CLASSES[y_input[i].item()])
            )
        plt.tight_layout()
        plt.show()
