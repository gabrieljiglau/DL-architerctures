import os

import torch
import torchvision.models as models
from torchvision.models import ResNet18_Weights
from utils import load_cifar, train_net, validate_net


def transfer_learning_setup(model, num_classes, freeze_extraction_layers=True):

    if freeze_extraction_layers:
        for param in model.parameters():
            param.requires_grad = False

    num_features = model.fc.in_features
    model.fc = torch.nn.Linear(num_features, num_classes)

    return model

if __name__ == '__main__':
    # TensorDataset class from pytorch

    # TODO: i) plots with matplotlib
    # TODO: ii) transfer learning + decision trees / logistic regression
    resnet18 = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    resnet18 = transfer_learning_setup(resnet18, 10)

    trainable_params = sum(p.numel() for p in resnet18.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in resnet18.parameters())

    print(f"trainable_parameters = {trainable_params}")
    print(f"total_params = {total_params}")

    train_loader, valid_loader, test_loader, classes = load_cifar(batch_size=64, num_workers=2)
    model_path = 'models/resnet18_pretrained.pt'
    optim = torch.optim.Adam(
        filter(lambda p: p.requires_grad, resnet18.parameters()),
        lr=1e-5
    )

    # 45 % accuracy : (
    if not os.path.exists(model_path):
        train_net(resnet18, 20, train_loader, valid_loader, optim, torch.nn.CrossEntropyLoss(), model_path)

    # idee: antrenezi numai ultimul strat pentru cateva generatii, apoi antrenezi toata reteaua

