import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    # This defines the structure of the NN.
    def __init__(self, dropout_percentage=0, norm="bn", num_groups=2, padding=1):
        super(Net, self).__init__()

        if norm == "bn":
            self.norm = nn.BatchNorm2d
        elif norm == "gn":
            self.norm = lambda in_dim: nn.GroupNorm(
                num_groups=num_groups, num_channels=in_dim
            )
        elif norm == "ln":
            self.norm = lambda in_dim: nn.GroupNorm(num_groups=1, num_channels=in_dim)

        # This defines the structure of the NN.

        # Prep Layer

        self.prep_layer = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),  # 32x32x3 | 1 -> 32x32x64 | 3
            self.norm(64),
            nn.ReLU(),
            nn.Dropout(dropout_percentage),
        )

        self.l1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # 32x32x128 | 5
            nn.MaxPool2d(2, 2),  # 16x16x128 | 6
            self.norm(128),
            nn.ReLU(),
            nn.Dropout(dropout_percentage),
        )
        self.l1res = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),  # 16x16x128 | 10
            self.norm(128),
            nn.ReLU(),
            nn.Dropout(dropout_percentage),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),  # 16x16x128 | 14
            self.norm(128),
            nn.ReLU(),
            nn.Dropout(dropout_percentage),
        )
        self.l2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),  # 16x16x256 | 18
            nn.MaxPool2d(2, 2),  # 8x8x256 | 19
            self.norm(256),
            nn.ReLU(),
            nn.Dropout(dropout_percentage),
        )
        self.l3 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),  # 8x8x512 | 27
            nn.MaxPool2d(2, 2),  # 4x4x512 | 28
            self.norm(512),
            nn.ReLU(),
            nn.Dropout(dropout_percentage),
        )
        self.l3res = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),  # 4x4x512 | 36
            self.norm(512),
            nn.ReLU(),
            nn.Dropout(dropout_percentage),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),  # 4x4x512 | 44
            self.norm(512),
            nn.ReLU(),
            nn.Dropout(dropout_percentage),
        )
        self.maxpool = nn.MaxPool2d(4, 4)

        # Classifier
        self.lr = nn.Linear(512, 10)

    def forward(self, x):
        x = self.prep_layer(x)
        x = self.l1(x)
        x = x + self.l1res(x)
        x = self.l2(x)
        x = self.l3(x)
        x = x + self.l3res(x)
        x = self.maxpool(x)
        x = x.view(-1, 512)
        x = self.lr(x)
        return F.log_softmax(x, dim=1)
