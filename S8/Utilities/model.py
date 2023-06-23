import torch.nn as nn
import torch.nn.functional as F


class Net1(nn.Module):
    # This defines the structure of the NN.
    def __init__(self, dropout_percentage=0, norm="bn", num_groups=2, padding=1):
        super(Net1, self).__init__()

        if norm == "bn":
            self.norm = nn.BatchNorm2d
        elif norm == "gn":
            self.norm = lambda in_dim: nn.GroupNorm(
                num_groups=num_groups, num_channels=in_dim
            )
        elif norm == "ln":
            self.norm = lambda in_dim: nn.GroupNorm(num_groups=1, num_channels=in_dim)

        channel_2 = 16
        channel_3 = 32
        self.Conv1 = nn.Sequential(
            nn.Conv2d(3, channel_2, kernel_size=3, padding=padding),
            self.norm(channel_2),
            nn.ReLU(),
            nn.Dropout(dropout_percentage),
            nn.Conv2d(channel_2, channel_3, kernel_size=3, padding=padding),
            self.norm(channel_3),
            nn.ReLU(),
            nn.Dropout(dropout_percentage),
        )
        channel_4 = 16
        self.conv2 = nn.Sequential(
            nn.Conv2d(channel_3, channel_4, kernel_size=1), nn.MaxPool2d(2, 2)
        )
        channel_5 = 16
        channel_6 = 32
        channel_7 = 32
        self.Conv3 = nn.Sequential(
            nn.Conv2d(channel_4, channel_5, kernel_size=3, padding=padding),
            self.norm(channel_5),
            nn.ReLU(),
            nn.Dropout(dropout_percentage),
            nn.Conv2d(channel_5, channel_6, kernel_size=3, padding=padding),
            self.norm(channel_6),
            nn.ReLU(),
            nn.Dropout(dropout_percentage),
            nn.Conv2d(channel_6, channel_7, kernel_size=3, padding=padding),
        )
        self.res1 = nn.Conv2d(channel_4, channel_7, kernel_size=1)
        self.conv4 = nn.Sequential(
            self.norm(channel_7),
            nn.ReLU(),
            nn.Dropout(dropout_percentage),
            nn.Conv2d(channel_7, channel_4, kernel_size=1),
            nn.MaxPool2d(2, 2),
        )
        self.Conv5 = nn.Sequential(
            nn.Conv2d(channel_4, channel_5, kernel_size=3, padding=padding),
            self.norm(channel_5),
            nn.ReLU(),
            nn.Dropout(dropout_percentage),
            nn.Conv2d(channel_5, channel_6, kernel_size=3, padding=padding),
            self.norm(channel_6),
            nn.ReLU(),
            nn.Dropout(dropout_percentage),
            nn.Conv2d(channel_6, channel_7, kernel_size=3, padding=padding),
        )
        self.res2 = nn.Conv2d(channel_4, channel_7, kernel_size=1)
        self.avgpool = nn.Sequential(
            self.norm(channel_7),
            nn.ReLU(),
            nn.Dropout(dropout_percentage),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channel_7, 10, kernel_size=1),
        )

    def forward(self, x):
        x = self.Conv1(x)
        x = self.conv2(x)
        x = self.Conv3(x) + self.res1(x)
        x = self.conv4(x)
        x = self.Conv5(x) + self.res2(x)
        x = self.avgpool(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=1)
