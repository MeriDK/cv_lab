import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from torchvision import transforms
import torchvision.models as models
import matplotlib.pyplot as plt
from tqdm import tqdm
import timm
import numpy as np
from torch.utils.data import random_split
from torch.utils.tensorboard import SummaryWriter


class Model(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super(Model, self).__init__()

        # Load a pretrained resnet model from torchvision.models in Pytorch
        self.model = timm.create_model('inception_resnet_v2', pretrained=False)

        # Change the input layer to take Grayscale image, instead of RGB images.
        # Hence in_channels is set as 1 or 3 respectively
        # original definition of the first layer
        # self.conv1 = Conv2d(3, 32, kernel_size=(3, 3), stride=(2, 2), bias=False)
        self.model.conv2d_1a.conv = nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, bias=False)

        # Change the output layer to output 10 classes instead of 1000 classes
        num_ftrs = self.model.classif.in_features
        self.model.classif = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.model(x)


def train(dataloader, model, loss_fn, optimizer, device):
    model.train()

    for batch, (X, y) in enumerate(tqdm(dataloader)):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def test(dataloader, model, loss_fn, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in tqdm(dataloader):
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size

    return correct, test_loss


def main():
    transform = transforms.Compose([
        transforms.Resize(139),  # minimum image size for inception resnet
        transforms.ToTensor(),
    ])

    # Download training data from open datasets.
    training_data = datasets.FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=transform,
    )

    # Download test data from open datasets.
    test_data = datasets.FashionMNIST(
        root="data",
        train=False,
        download=True,
        transform=transform,
    )

    # Separate train dataset to train and val
    train_len = int(0.8 * len(training_data))
    val_len = len(training_data) - train_len
    torch.manual_seed(42)
    train_data, val_data = random_split(training_data, [train_len, val_len])

    batch_size = 128

    # Create data loaders.
    train_dataloader = DataLoader(train_data, batch_size=batch_size)
    val_dataloader = DataLoader(val_data, batch_size=batch_size)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)

    # Get cpu or gpu device for training.
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = Model()
    model = model.to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    writer = SummaryWriter(log_dir='results')

    epochs = 20

    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer, device)

        train_correct, train_loss = test(train_dataloader, model, loss_fn, device)
        val_correct, val_loss = test(val_dataloader, model, loss_fn, device)

        print(f"\nTrain: Accuracy: {(100 * train_correct):>0.1f}%, Avg loss: {train_loss:>8f}")
        print(f"\nVal: Accuracy: {(100 * val_correct):>0.1f}%, Avg loss: {val_loss:>8f}")

        writer.add_scalars("running/epoch_loss", {"train": train_loss, "val": val_loss}, t)
        writer.add_scalars("running/accuracy", {"train": train_correct, "val": val_correct}, t)

        torch.save(model.state_dict(), f"results/model{t}.pth")

    test_correct, _ = test(test_dataloader, model, loss_fn, device)
    print(f"Test: \n Accuracy: {(100 * test_correct):>0.1f}%\n")

    print("Done!")


if __name__ == '__main__':
    main()
