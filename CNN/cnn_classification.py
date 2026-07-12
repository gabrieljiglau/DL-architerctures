import os

import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
from utils import load_cifar, train_net, validate_net

class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ConvBlock, self).__init__()

        self.block = nn.Sequential(
            # instead of using MaxPoll2d, which is fixed parameter, one can discard it for a learnable parameter,
            # by using stride=2, in Conv2d
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.block(x)



class CNN(nn.Module):
    def __init__(self, channels, strides, paddings, kernels=None, num_classes=10):
        super(CNN, self).__init__()

        self.layers = nn.ModuleList(
            [ConvBlock(in_channels=in_channel, out_channels=out_channel,
                       kernel_size=kernels[index], stride=strides[index], padding=paddings[index])
             for index, (in_channel, out_channel) in enumerate(zip(channels[:-1], channels[1:]))]
        )

        # by default a cifar image is 32 X 32, and we have 3 convolution layers with stride 2
        # (meaning downsampling by 2 each dimension)
        # therefore, now the size of the input is (32 / 16 = 2) X (32 / 16 = 2)
        # and there are channels[-1] channels, as specified from the last convolution

        self.fc1 = nn.Linear(channels[-1] * 4 * 4, 1028)
        self.fc2 = nn.Linear(1028, 512)
        self.fc3 = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):

        # print(f"shape(x) = {np.shape(x)}")
        for layer in self.layers:
            x = layer(x)
            # print(f"shape(x) = {x.size()}")

        x = torch.flatten(x, 1) # flatten everything, starting from dim=1, in order to prepare the data for the fc layer
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.fc3(x)

        return x


if __name__ == '__main__':

    kernels = [3] * 5
    strides = [2, 2, 1, 1, 2]
    paddings = [1] * 5

    """
    Training accuracy 73.885% | Validation accuracy 71.000%
    Validation loss decreased (0.8592 ---> 0.8455). Saving model as net_cifar10.pt
    
    Test loss: 1.045609
    Test accuracy = 64.44%
    """

    # channels[0] = how many channels the original image have;

    # increase the out_channels after each convolution because of what is happening simultaneously:
    # i) spatial resolution (height × width) is decreasing.
    # ii) feature complexity is increasing.
    channels = [3, 96, 256, 384, 384, 256]

    net = CNN(channels=channels, strides=strides, paddings=paddings, kernels=kernels, num_classes=10)
    optimizer = optim.Adam(net.parameters(), lr=1e-5)

    train_loader, valid_loader, test_loader, classes = load_cifar(batch_size=64, num_workers=2)

    if not os.path.exists('models/net_cifar10.pt'):
        train_net(net, num_epochs=20, train_loader=train_loader, valid_loader=valid_loader, optimizer=optimizer,
                  loss_metric=nn.CrossEntropyLoss(), model_path='models/net_cifar10.pt')

    if os.path.exists('models/net_cifar10.pt'):
        validate_net(net, test_loader, 'models/net_cifar10.pt', nn.CrossEntropyLoss())















        