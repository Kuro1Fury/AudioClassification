import torch
import os
import soundfile as sf
import torchaudio
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import pandas as pd
from datasets import Dataset, Audio


# 1- download dataset
# 2- create data loader
# 3- build model
# 4- train
# 5- save trained model


BATCH_SIZE = 128
EPOCHS = 10
LEARNING_RATE = 0.001


class FeedForwardNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.dense_layers = nn.Sequential(
            nn.Linear(28 * 28, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_data):
        x = self.flatten(input_data)
        logits = self.dense_layers(x)
        predictions = self.softmax(logits)
        return predictions


def download_mnist_datasets():
    train_data = datasets.MNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor(),
    )
    validation_data = datasets.MNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor(),
    )
    return train_data, validation_data


def create_data_loader(train_data, batch_size):
    train_dataloader = DataLoader(train_data, batch_size=batch_size)
    return train_dataloader


def train_single_epoch(model, data_loader, loss_fn, optimiser, device):
    for input, target in data_loader:
        input, target = input.to(device), target.to(device)

        # calculate loss
        prediction = model(input)
        loss = loss_fn(prediction, target)

        # backpropagate error and update weights
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

    print(f"loss: {loss.item()}")


def train(model, data_loader, loss_fn, optimiser, device, epochs):
    for i in range(epochs):
        print(f"Epoch {i+1}")
        train_single_epoch(model, data_loader, loss_fn, optimiser, device)
        print("---------------------------")
    print("Finished training")


class BirdSoundsDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.bird_folders = os.listdir(data_dir)

    def __len__(self):
        return len(self.bird_folders)

    def __getitem__(self, idx):
        bird_folder = self.bird_folders[idx]
        bird_folder_path = os.path.join(self.data_dir, bird_folder)
        wav_files = os.listdir(bird_folder_path)
        bird_samples = []

        for wav_file in wav_files:
            wav_path = os.path.join(bird_folder_path, wav_file)
            waveform, sample_rate = torchaudio.load(wav_path)
            bird_samples.append(waveform)

        return bird_samples

if __name__ == "__main__":

    # download data and create data loader
    # train_data, _ = download_mnist_datasets()
    # train_dataloader = create_data_loader(train_data, BATCH_SIZE)

    data = pd.read_csv("bird_dataset.csv")
    paths = data['Path'].tolist()
    # append 'dataset/' to each path
    paths = ['dataset/' + path for path in paths]

    bird_dataset = Dataset.from_dict({"audio": paths}).cast_column("audio", Audio())
    train_dataloader = DataLoader(bird_dataset, batch_size=32, shuffle=True)

    # construct model and assign it to device
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using {device}")
    feed_forward_net = FeedForwardNet().to(device)
    print(feed_forward_net)

    # initialise loss funtion + optimiser
    loss_fn = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(feed_forward_net.parameters(),
                                 lr=LEARNING_RATE)

    # train model
    train(feed_forward_net, train_dataloader, loss_fn, optimiser, device, EPOCHS)

    # save model
    torch.save(feed_forward_net.state_dict(), "feedforwardnet.pth")
    print("Trained feed forward net saved at feedforwardnet.pth")