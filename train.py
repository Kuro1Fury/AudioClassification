import numpy as np
import torch
import torchaudio
from torch import nn
from torch.utils.data import DataLoader, SubsetRandomSampler
from sklearn.model_selection import train_test_split
from skorch import NeuralNetClassifier
from sklearn.model_selection import GridSearchCV

from BirdDataLoader import BirdSoundDataset
from cnn import CNNNetwork

BATCH_SIZE = 128
EPOCHS = 10
LEARNING_RATE = 0.001
VALIDATION_SPLIT = 0.2
TEST_SPLIT = 0.1

ANNOTATIONS_FILE = "metadata/bird_dataset_full.csv"
AUDIO_DIR = "dataset/Audio"
SAMPLE_RATE = 22050
NUM_SAMPLES = 22050

def create_data_loader(data, batch_size):
    data_loader = DataLoader(data, batch_size=batch_size)
    return data_loader


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

def evaluate(model, data_loader, loss_fn, device):
    size = len(data_loader.dataset)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in data_loader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= size
    correct /= size
    print(f"Validation Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


def train(model, train_dataloader, val_dataloader, loss_fn, optimiser, device, epochs):
    for i in range(epochs):
        print(f"Epoch {i + 1}")
        train_single_epoch(model, train_dataloader, loss_fn, optimiser, device)
        evaluate(model, val_dataloader, loss_fn, device)
        print("---------------------------")
    print("Finished training")


if __name__ == "__main__":
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using {device}")

    # instantiating our dataset object and create data loader
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=1024,
        hop_length=512,
        n_mels=64
    )

    bsd = BirdSoundDataset(ANNOTATIONS_FILE,
                            AUDIO_DIR,
                            mel_spectrogram,
                            SAMPLE_RATE,
                            NUM_SAMPLES,
                            device)


    dataset_size = len(bsd)
    indices = list(range(dataset_size))
    val_split = int(np.floor(VALIDATION_SPLIT * dataset_size))
    test_split = int(np.floor(TEST_SPLIT * dataset_size))

    np.random.shuffle(indices)
    train_indices, val_indices, test_indices = indices[val_split + test_split:], indices[:val_split], indices[
                                                                                                      val_split:val_split + test_split]

    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    train_dataloader = DataLoader(bsd, batch_size=BATCH_SIZE, sampler=train_sampler)
    val_dataloader = DataLoader(bsd, batch_size=BATCH_SIZE, sampler=val_sampler)
    test_dataloader = DataLoader(bsd, batch_size=BATCH_SIZE, sampler=test_sampler)

    best_model = None
    best_acc = 0.0
    best_params = None

    batch_sizes = [64, 128]
    lrs = [0.0001, 0.0001, 0.001]
    epochs = [20]
    for lr in lrs:
        for batch_size in batch_sizes:
            for epoch in epochs:
                X_train, y_train = next(iter(train_dataloader))
                X_val, y_val = next(iter(val_dataloader))

                X_train = X_train.numpy()
                y_train = y_train.numpy()
                X_val = X_val.numpy()
                y_val = y_val.numpy()

                model = NeuralNetClassifier(
                    CNNNetwork,
                    max_epochs=epoch,
                    lr=lr,
                    batch_size=batch_size,
                    optimizer=torch.optim.Adam,
                    criterion=nn.CrossEntropyLoss,
                    device=device
                )
                model.fit(X_train, y=y_train)
                acc = model.score(X_val, y=y_val)
                print("using lr: ", lr, " batch size: ", batch_size, " epoch: ", epoch, " acc: ", acc)
                print("Accuracy: ", acc)
                if acc > best_acc:
                    best_acc = acc
                    best_model = model
                    best_params = (lr, batch_size, epoch)

    print("Best accuracy: ", best_acc)
    print("Best parameters: ", best_params)
    print("Best model: ", best_model)