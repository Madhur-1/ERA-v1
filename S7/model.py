import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    # This defines the structure of the NN.
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm2d(10),
            nn.Conv2d(10, 10, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm2d(10),
            nn.Conv2d(10, 20, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm2d(20),

        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(20, 10, kernel_size=1),
            # nn.ReLU(),
            # nn.BatchNorm2d(10),
            nn.Conv2d(10, 10, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm2d(10),
            nn.Conv2d(10, 10, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm2d(10),
            nn.Conv2d(10, 10, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm2d(10),
        )
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        # self.fc1 = nn.Linear(32, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pool2d(x, 2)
        x = self.conv2(x)
        x = self.avgpool(x)
        x = x.view(-1, 10)
        # x = self.fc1(x)
        return F.log_softmax(x, dim=1)


class Net1(nn.Module):
    # This defines the structure of the NN.
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 16, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 32, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm2d(32),

        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=1, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 16, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 16, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 32, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm2d(32),
        )
        self.avgpool = nn.AvgPool2d(5)
        self.fc1 = nn.Linear(32, 10)

    def forward(self, x):
        x = self.conv1(x)
        # x = F.relu(F.max_pool2d(x, 2))
        x = self.conv2(x)
        x = self.avgpool(x)
        x = x.view(-1, 32)
        x = self.fc1(x)
        return F.log_softmax(x, dim=1)
