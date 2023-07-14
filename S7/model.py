import torch.nn as nn
import torch.nn.functional as F


class Net6(nn.Module):
    # This defines the structure of the NN.
    def __init__(self):
        super(Net6, self).__init__()

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
        return F.log_softmax(x, dim=1)


class Net5(nn.Module):
    # This defines the structure of the NN.
    def __init__(self):
        super(Net5, self).__init__()

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
            nn.Conv2d(10, 28, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm2d(28),
            nn.Conv2d(28, 10, kernel_size=1),
        )
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pool2d(x, 2)
        x = self.conv2(x)
        x = self.avgpool(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=1)


class Net4(nn.Module):
    # This defines the structure of the NN.
    def __init__(self):
        super(Net4, self).__init__()

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
            nn.Conv2d(10, 20, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm2d(20),
            nn.Conv2d(20, 10, kernel_size=1),
        )
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pool2d(x, 2)
        x = self.conv2(x)
        x = self.avgpool(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=1)


class Net3(nn.Module):
    # This defines the structure of the NN.
    def __init__(self):
        super(Net3, self).__init__()

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
            nn.Conv2d(10, 20, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm2d(20),
            nn.Conv2d(20, 10, kernel_size=1),
        )
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pool2d(x, 2)
        x = self.conv2(x)
        x = self.avgpool(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=1)


class Net2(nn.Module):
    # This defines the structure of the NN.
    def __init__(self):
        super(Net2, self).__init__()

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

    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pool2d(x, 2)
        x = self.conv2(x)
        x = self.avgpool(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=1)


class Net1(nn.Module):
    # This defines the structure of the NN.
    def __init__(self):
        super(Net1, self).__init__()

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
            # nn.ReLU(),
            # nn.BatchNorm2d(32),
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
        x = self.conv2(x)
        x = self.avgpool(x)
        x = x.view(-1, 32)
        x = self.fc1(x)
        return F.log_softmax(x, dim=1)
