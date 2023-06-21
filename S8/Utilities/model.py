import torch.nn as nn
import torch.nn.functional as F


class Net1(nn.Module):
    # This defines the structure of the NN.
    def __init__(self, use_dropout=False, norm="bn", num_groups=2):
        super(Net1, self).__init__()

        if norm == "bn":
            self.norm = nn.BatchNorm2d
        elif norm == "gn":
            self.norm = lambda in_dim: nn.GroupNorm(
                num_groups=num_groups, num_channels=in_dim
            )
        elif norm == "ln":
            self.norm = nn.LayerNorm

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=3),
            nn.ReLU(),
            self.norm(10),
            nn.Conv2d(10, 10, kernel_size=3),
            nn.ReLU(),
            self.norm(10),
            nn.Conv2d(10, 11, kernel_size=3),
            nn.ReLU(),
            self.norm(11),
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
        return F.log_softmax(x, dim=1)
