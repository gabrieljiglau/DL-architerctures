import os.path
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from utils import load_cifar, train_net

class ResidualBlock(nn.Module):

    expansion = 1  # output channels multiplier

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
        # print(f"np.shape(x) = {np.shape(x)}")
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

        self.avgPool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128 * block.expansion, num_classes)


    def _make_layer(self, block:ResidualBlock, out_channels, blocks: int, stride=1):

        """
        :param block: type of residual block to be used
        :param out_channels: the number of output channels
        :param blocks: how many blocks to be used in this layer
        :param stride: stride for the first block (for downsampling)
        :return: nn.Sequential(*layers)
        """

        downsample = None

        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion)
            )

        #   def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, downsample=None):
        layers = [block(self.in_channels, out_channels, stride=stride, downsample=downsample)]

        # first block may need downsampling
        self.in_channels = out_channels * block.expansion

        for _ in range(1, blocks):
            layers.append((block(self.in_channels, out_channels)))

        return nn.Sequential(*layers)


    def forward(self, x):

        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgPool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


if __name__ == '__main__':

    # TensorDataset class from pytorch

    # TODO: i) plots with matplotlib
    # TODO: ii) transfer learning + decision trees / logistic regression

    train_loader, valid_loader, test_loader, num_classes = load_cifar(batch_size=64, num_workers=2)
    # num_classes = ['plane', 'vehicle', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'] !!

    resnet8 = ResidualNetwork(ResidualBlock, [2, 2, 2], len(num_classes))
    optim = torch.optim.Adam(resnet8.parameters(), lr=1e-5)
    model_path = 'models/resnet_cifar10.pt'

    """
    x = torch.randn(1, 3, 224, 224)
    output = model(x)
    print(f"output.shape = {np.shape(output)}\noutput = {output}")
    """


    if not os.path.exists(model_path):
        train_net(resnet8, num_epochs=20, train_loader=train_loader, valid_loader=valid_loader, optimizer=optim,
                  loss_metric=nn.CrossEntropyLoss(), model_path=model_path)

