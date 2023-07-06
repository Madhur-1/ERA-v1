import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
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
            nn.Conv2d(16, 30, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm2d(30),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(30, 30, kernel_size=1, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(30),
            nn.Conv2d(30, 16, kernel_size=3),
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
