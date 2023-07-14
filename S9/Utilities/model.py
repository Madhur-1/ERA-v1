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

        channel_2 = 24
        channel_3 = 44
        channel_4 = 64

        self.C1c1 = nn.Sequential(
            nn.Conv2d(
                3, channel_2, kernel_size=3, padding=padding
            ),  # 32x32x3 | 1 -> 32x32xchannel_2 | 3
            self.norm(channel_2),
            nn.ReLU(),
            nn.Dropout(dropout_percentage),
        )
        self.C1c2 = nn.Sequential(
            nn.Conv2d(
                channel_2, channel_3, kernel_size=3, padding=padding
            ),  # 32x32xchannel_3 | 5
            self.norm(channel_3),
            nn.ReLU(),
            nn.Dropout(dropout_percentage),
        )
        self.C1c3 = nn.Sequential(
            nn.Conv2d(
                channel_3, channel_4, kernel_size=3, padding=padding, stride=2
            ),  # 16x16xchannel_4 | 7
            self.norm(channel_4),
            nn.ReLU(),
            nn.Dropout(dropout_percentage),
        )

        self.res1 = nn.Conv2d(channel_4, channel_3, kernel_size=1)

        self.C2c1 = nn.Sequential(
            nn.Conv2d(
                channel_4, channel_2, kernel_size=3, padding=padding
            ),  # 16x16xchannel_2 | 11
            self.norm(channel_2),
            nn.ReLU(),
            nn.Dropout(dropout_percentage),
        )
        self.C2c2 = nn.Sequential(
            nn.Conv2d(
                channel_2, channel_3, kernel_size=3, padding=padding
            ),  # 16x16xchannel_3 | 15
            self.norm(channel_3),
            nn.ReLU(),
            nn.Dropout(dropout_percentage),
        )
        self.C2c3 = nn.Sequential(
            nn.Conv2d(
                channel_3, channel_4, kernel_size=3, padding=padding, stride=2
            ),  # 8x8xchannel_4 | 19
            self.norm(channel_4),
            nn.ReLU(),
            nn.Dropout(dropout_percentage),
        )
        self.res2 = nn.Conv2d(channel_4, channel_3, kernel_size=1)

        self.C3c1 = nn.Sequential(
            nn.Conv2d(
                channel_4, channel_2, kernel_size=3, padding=padding
            ),  # 8x8xchannel_2 | 27
            self.norm(channel_2),
            nn.ReLU(),
            nn.Dropout(dropout_percentage),
        )
        self.C3c2 = nn.Sequential(
            nn.Conv2d(
                channel_2, channel_3, kernel_size=3, padding=padding
            ),  # 8x8xchannel_3 | 35
            self.norm(channel_3),
            nn.ReLU(),
            nn.Dropout(dropout_percentage),
        )
        self.C3c3 = nn.Sequential(
            nn.Conv2d(
                channel_3, channel_4, kernel_size=3, padding=padding, dilation=2
            ),  # 6x6xchannel_4 | 51
            self.norm(channel_4),
            nn.ReLU(),
            nn.Dropout(dropout_percentage),
        )
        self.res3 = nn.Conv2d(channel_4, channel_3, kernel_size=1)

        self.C4c1 = nn.Sequential(
            nn.Conv2d(
                channel_4,
                channel_4 * 2,
                kernel_size=3,
                padding=padding,
                groups=channel_4,
            ),  # 6x6xchannel_4 | 59
            nn.Conv2d(channel_4 * 2, channel_2, kernel_size=1),
            self.norm(channel_2),
            nn.ReLU(),
            nn.Dropout(dropout_percentage),
        )
        self.C4c2 = nn.Sequential(
            nn.Conv2d(
                channel_2,
                channel_2 * 2,
                kernel_size=3,
                padding=padding,
                groups=channel_2,
            ),  # 6x6xchannel_2*2 | 67
            nn.Conv2d(channel_2 * 2, channel_3, kernel_size=1),
            self.norm(channel_3),
            nn.ReLU(),
            nn.Dropout(dropout_percentage),
        )
        self.C4c3 = nn.Sequential(
            nn.Conv2d(
                channel_3,
                channel_3 * 2,
                kernel_size=3,
                padding=padding,
                groups=channel_3,
            ),  # 6x6xchannel_3*2 | 75
            nn.Conv2d(channel_3 * 2, channel_4, kernel_size=1),
            self.norm(channel_4),
            nn.ReLU(),
            nn.Dropout(dropout_percentage),
        )

        self.out = nn.Sequential(
            nn.AdaptiveMaxPool2d(1),
            nn.Conv2d(channel_4, 10, kernel_size=1),
        )

    def forward(self, x):
        x = self.C1c1(x)
        x = self.C1c2(x)
        x = self.C1c3(x)
        C1c3 = x
        x = self.C2c1(x)
        x = self.C2c2(x) + self.res1(C1c3)
        x = self.C2c3(x)
        C2c3 = x
        x = self.C3c1(x)
        x = self.C3c2(x) + self.res2(C2c3)
        x = self.C3c3(x)
        C3c3 = x
        x = self.C4c1(x)
        x = self.C4c2(x) + self.res3(C3c3)
        x = self.C4c3(x)
        x = self.out(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=1)
