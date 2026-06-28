import numpy as np
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split

def load_cifar(batch_size=64, num_workers=4):

    transform = transforms.Compose([
        transforms.ToTensor(),
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