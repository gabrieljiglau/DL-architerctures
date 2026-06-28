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

        # training
        net.train()
        for data, target in train_loader:
            data, target = data, target
            output = net(data)
            loss = loss_metric(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * data.size(0)

        # validation
        net.eval()
        with torch.no_grad():
            for data, target in valid_loader:
                data, target = data, target
                output = net(data)
                loss = loss_metric(output, target)
                valid_loss += loss.item() * data.size(0)

        train_loss /= len(train_loader.dataset)
        valid_loss /= len(valid_loader.dataset)

        end_time = time.time()
        epoch_time = end_time - start_time

        print(f"Epoch {epoch + 1}/{num_epochs} | Time: {epoch_time:.3f}s | Training loss: {train_loss:.4f} | "
              f"Validation loss: {valid_loss:.4f}")

        if valid_loss <= min_loss:
            print(f"Validation loss decreased ({min_loss:.4f} ---> {valid_loss:.4f}). Saving model as net_cifar10.pt")
            torch.save(net.state_dict(),'models/net_cifar10.pt')
            valid_loss = min_loss
            break



class CNN(nn.Module):
    def __init__(self, lr, num_channels=3, num_classes=10):
        super(CNN, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=num_channels, out_channels=32, kernel_size=3, stride=1, padding=1), # stride automatically is 1
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
            # instead of using MaxPoll2d, fixed parameter, one can discard it and instead use stride=2, in Conv2d
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )

        self.fc1 = nn.Linear(16 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = torch.flatten(x, 1) # flatten everything, starting from dim=1
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)

        return x


# TODO: i) add validation results;
# TODO: ii) plot some images with matplotlib;
# TODO: iii) change the fully connected layers with softmax regression





if __name__ == '__main__':

    net = CNN(lr=1e-4, num_channels=3, num_classes=10)
    optimizer = optim.Adam(net.parameters(), lr=1e-4)

    train_loader, valid_loader, test_loader, classes = load_cifar(batch_size=64, num_workers=2)
    train(net, num_epochs=20, train_loader=train_loader, valid_loader=valid_loader, optimizer=optimizer, loss_metric=nn.CrossEntropyLoss())















        