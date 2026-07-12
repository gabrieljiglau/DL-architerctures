import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.init import kaiming_normal

from utils import load_cifar, train_net

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

        # no bias if we use batch normalization right after
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        # store input for the skip connection
        identity = x

        out = self.block1(x)
        out = F.relu(out)

        out = self.block2(out)
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        return F.relu(out)


class ResidualNetwork(nn.Module):
    def __init__(self, block:ResidualBlock, layers: list[int], num_classes=10):
        super(ResidualNetwork, self).__init__()

        self.in_channels = 32
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.layer1 = self._make_layer(block, 32, layers[0])
        self.layer2 = self._make_layer(block, 64, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 128, layers[2], stride=2)

        self.avgPool = nn.AdaptiveAvgPool1d((1, 1))
        self.fc = nn.Linear(128 * block.expansion, num_classes)


    def _make_layer(self, block:ResidualBlock, in_channels, blocks, stride=1):

        downsample = None

        if stride != 1 or self.in_channels != in_channels * blocks.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, in_channels * blocks.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(in_channels * blocks.expansion)
            )

        layers = []

        # first block may need downsampling
        layers.append(block(self.in_channels, in_channels, stride=stride, downsample=downsample))
        self.in_channels = in_channels * blocks.expansion

        for _ in range(1, blocks):
            layers.append((block(self.in_channels, in_channels)))

        return nn.Sequential(*layers)


    def forward(self, x):

        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgPool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)

def resnet(num_classes):
    return ResidualNetwork(ResidualBlock, [2, 2, 2], num_classes)


if __name__ == '__main__':

    # TODO: i) plots with matplotlib
    # TODO: ii) representation learning + decision trees / logistic regression

    model = resnet(10)
