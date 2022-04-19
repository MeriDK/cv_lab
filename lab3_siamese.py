import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import numpy as np
import random
from torch.utils.tensorboard import SummaryWriter
import timm
from torch.utils.data import random_split
from torchvision import transforms
from torchvision import datasets


class SiameseDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, index):
        img1, label1 = random.choice(self.dataset)

        # We need to approximately 50% of images to be in the same class
        should_get_same_class = random.randint(0, 1)
        if should_get_same_class:
            while True:
                # Look untill the same class image is found
                img2, label2 = random.choice(self.dataset)
                if label1 == label2:
                    break
        else:
            while True:
                # Look untill a different class image is found
                img2, label2 = random.choice(self.dataset)
                if label1 != label2:
                    break

        return img1, img2, torch.from_numpy(np.array([int(label1 != label2)], dtype=np.float32))

    def __len__(self):
        return len(self.dataset)


# create the Siamese Neural Network
# create the Siamese Neural Network
class SiameseNetwork(nn.Module):

    def __init__(self):
        super(SiameseNetwork, self).__init__()

        # Setting up the Fully Connected Layers
        self.fc = nn.Sequential(
            nn.Linear(1536, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, input1, input2):
        combined_features = input1 * input2

        return self.fc(combined_features)


class FeatureModel(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super(FeatureModel, self).__init__()

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


def train(dataloader, model, feature_model, loss_fn, optimizer, device):
    model.train()

    for batch, (img0, img1, label) in enumerate(tqdm(dataloader)):

        # Send the images and labels to CUDA
        img0, img1, label = img0.to(device), img1.to(device), label.to(device)

        img0, img1 = feature_model(img0), feature_model(img1)

        # Pass in the two images into the network
        prob = model(img0, img1)

        # Pass the output of the networks and label into the loss function
        loss = loss_fn(prob, label)

        # Zero the gradients
        optimizer.zero_grad()

        # Calculate the backpropagation
        loss.backward()

        # Optimize
        optimizer.step()


def test(dataloader, model, feature_model, loss_fn, device):
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0

    with torch.no_grad():
        for img0, img1, label in tqdm(dataloader):
            img0, img1, label = img0.to(device), img1.to(device), label.to(device)
            img0, img1 = feature_model(img0), feature_model(img1)

            prob = model(img0, img1)
            test_loss += loss_fn(prob, label).item()
            correct += torch.count_nonzero(label == (prob > 0.5)).item()

    test_loss /= num_batches
    correct /= len(dataloader.dataset)

    return test_loss, correct


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

    train_len = int(0.8 * len(training_data))
    val_len = len(training_data) - train_len
    torch.manual_seed(42)
    train_data, val_data = random_split(training_data, [train_len, val_len])

    train_datas = SiameseDataset(train_data)
    val_datas = SiameseDataset(val_data)
    test_datas = SiameseDataset(test_data)

    batch_size = 64

    train_dataloader = DataLoader(train_datas, batch_size=batch_size)
    val_dataloader = DataLoader(val_datas, batch_size=batch_size)
    test_dataloader = DataLoader(test_datas, batch_size=batch_size)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = SiameseNetwork()
    model = model.to(device)

    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

    feature_model = FeatureModel()
    feature_model.load_state_dict(torch.load('results/classifier/model19.pth'))
    feature_model.model.classif = nn.Identity()
    feature_model.to(device)
    feature_model.eval()

    epochs = 100
    writer = SummaryWriter(log_dir='results/siamese2')

    # Iterate through the epochs
    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train(train_dataloader, model, feature_model, criterion, optimizer, device)

        train_loss, train_correct = test(train_dataloader, model, feature_model, criterion, device)
        val_loss, val_correct = test(val_dataloader, model, feature_model, criterion, device)

        print(f"\nTrain Avg loss: {train_loss:>8f} Accuracy: {(100 * train_correct):>0.1f}%, ")
        print(f"\nVal Avg loss: {val_loss:>8f} Accuracy: {(100 * val_correct):>0.1f}%, ")

        writer.add_scalars("running/epoch_loss", {"train": train_loss, "val": val_loss}, t)
        writer.add_scalars("running/accuracy", {"train": train_correct, "val": val_correct}, t)

        torch.save(model.state_dict(), f"results/siamese2/model{t}.pth")

    _, test_correct = test(test_dataloader, model, feature_model, criterion, device)
    print(f"Test: \n Accuracy: {(100 * test_correct):>0.1f}%\n")

    print("Done!")


if __name__ == '__main__':
    main()
