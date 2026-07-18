import time
import numpy as np
import torchvision
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split

def load_cifar(batch_size=64, num_workers=4):

    """
    :return: train_loader, valid_loader, test_loader, classes
    """

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    full_train_data = torchvision.datasets.CIFAR10('data', train=True, download=True, transform=transform)
    test_data = torchvision.datasets.CIFAR10('data', train=False, download=True, transform=transform)

    num_train = len(full_train_data)
    split_size = int(np.floor(num_train * 0.2))
    train_size = num_train - split_size

    train_data, valid_data = random_split(full_train_data, [train_size, split_size])

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    classes = ['plane', 'vehicle', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    return train_loader, valid_loader, test_loader, classes

def train_net(net, num_epochs: int, train_loader:DataLoader, valid_loader:DataLoader,
          optimizer, loss_metric, model_path):

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
                validation_correct += (predicted == target).sum().item()
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
            print(f"Validation loss decreased ({min_loss:.4f} ---> {valid_loss:.4f}). Saving model as {model_path}")
            torch.save(net.state_dict(),model_path)
            min_loss = valid_loss


def validate_net(net, test_loader: DataLoader, model_path, loss_metric):
    net.load_state_dict(torch.load(model_path))

    test_loss = 0
    class_correct = [0] * 10
    class_total = [0] * 10

    net.eval()
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data, target
            output = net(data)
            loss = loss_metric(output, target)
            test_loss += loss.item() * data.size(0)
            _, y_pred = torch.max(output, 1)
            correct = y_pred.eq(target.view_as(y_pred))

            for i in range(len(target)):
                label = target[i].item()
                class_correct[label] += correct[i].item()
                class_total[label] += 1

    test_loss /= len(test_loader.dataset)
    print(f"Test loss: {test_loss:.6f}")

    accuracy = 100 * np.sum(class_correct) / np.sum(class_total)
    print(f"Test accuracy = {accuracy:.2f}%")