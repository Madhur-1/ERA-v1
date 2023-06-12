import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    # This defines the structure of the NN.
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3)
        self.bn3 = nn.BatchNorm2d(32)
        self.conv4 = nn.Conv2d(32, 32, kernel_size=3)
        self.bn4 = nn.BatchNorm2d(32)
        self.avgpool = nn.AvgPool2d(8)
        self.fc1 = nn.Linear(32, 10)

    def forward(self, x):
        x = self.bn1(F.relu(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.bn2(F.relu(self.conv2(x))), 2))
        x = self.bn3(F.relu(self.conv3(x), 2))
        x = self.bn4(F.relu(self.conv4(x)))
        x = self.avgpool(x)
        x = x.view(-1, 32)
        x = self.fc1(x)
        return F.log_softmax(x, dim=1)
