import time
import torch
import numpy as np
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from utils import load_cifar
from torch.utils.data import DataLoader


def train(net, num_epochs: int, train_loader:DataLoader, valid_loader:DataLoader,
          optimizer, loss_metric):

    min_loss = np.inf

    for epoch in range(num_epochs):
        start_time = time.time()
        train_loss = 0
        valid_loss = 0

        train_correct = 0
        train_total = 0

        # training
        net.train()
        for data, target in train_loader:
            data, target = data, target
            output = net(data)

            predicted = output.argmax(dim=1)
            train_correct += (predicted == target).sum().item()
            train_total += target.size(0) # the batch size

            loss = loss_metric(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * data.size(0)

        validation_correct = 0
        validation_total = 0

        # validation
        net.eval()
        with torch.no_grad():
            for data, target in valid_loader:
                data, target = data, target
                output = net(data)

                # dim = 1, the dimension that gets collapsed; since the size is (batch_size, num_classes)
                # that means it returns the max out of all the classes
                predicted = output.argmax(dim=1)
                validation_correct = (predicted == target).sum().item()
                validation_total += target.size(0)

                loss = loss_metric(output, target)
                valid_loss += loss.item() * data.size(0)

        train_loss /= len(train_loader.dataset)
        valid_loss /= len(valid_loader.dataset)

        end_time = time.time()
        epoch_time = end_time - start_time

        print(f"Epoch {epoch + 1}/{num_epochs} | Time: {epoch_time:.3f}s | Training loss: {train_loss:.4f} | "
              f"Validation loss: {valid_loss:.4f}")
        print(f"Training accuracy {(train_correct * 100 / train_total):.3f}% | "
              f"Validation accuracy {(validation_correct * 100 / validation_total):.3f}%")

        if valid_loss <= min_loss:
            print(f"Validation loss decreased ({min_loss:.4f} ---> {valid_loss:.4f}). Saving model as net_cifar10.pt")
            torch.save(net.state_dict(),'models/net_cifar10.pt')
            min_loss = valid_loss



class CNN(nn.Module):
    def __init__(self, num_channels=3, num_classes=10):
        super(CNN, self).__init__()

        """
        # poor performance with this current architecture !!!
        Epoch 10/20 | Time: 58.035s | Training loss: 1.4004 | Validation loss: 1.2875
        Training accuracy 49.008% | Validation accuracy 0.080%
        Validation loss decreased (1.3099 ---> 1.2875). Saving model as net_cifar10.pt

        """

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=num_channels, out_channels=32, kernel_size=3, stride=1, padding=1), # stride automatically is 1
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
            # instead of using MaxPoll2d, which is fixed parameter, one can discard it for a learnable parameter,
            # by using stride=2, in Conv2d
        )

        # increase the out_channels after each convolution because of what is happening simultaneously:
        # i) spatial resolution (height × width) is decreasing.
        # ii) feature complexity is increasing.
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )

        # by default a cifar image is 32 X 32, and we applied a pooling and a convolution layer with stride 2
        # therefore, now the size of the input is (32 / 2 / 2 = 8) X (32 / 2 / 2 = 8)
        # and there are 16 channels, as specified from the previous convolution
        self.fc1 = nn.Linear(16 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = torch.flatten(x, 1) # flatten everything, starting from dim=1, in order to prepare the data for the fc layer
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)

        return x


# TODO: i) add validation results;
# TODO: ii) plot some images with matplotlib;
# TODO: iii) change the fully connected layers with softmax regression





if __name__ == '__main__':

    net = CNN(num_channels=3, num_classes=10)
    optimizer = optim.Adam(net.parameters(), lr=1e-4)

    train_loader, valid_loader, test_loader, classes = load_cifar(batch_size=64, num_workers=2)
    train(net, num_epochs=20, train_loader=train_loader, valid_loader=valid_loader, optimizer=optimizer, loss_metric=nn.CrossEntropyLoss())















        