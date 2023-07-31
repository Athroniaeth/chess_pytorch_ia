import torch
import torch.nn as nn
import torch.nn.functional as functional
from torch import optim
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.models.detection import transform
from torchvision.transforms import transforms

import numpy as np


def create_random_array():
    # On crée un array contenant un 1
    arr = np.array([1])

    # On ajoute les zéros pour atteindre une taille de 64
    arr = np.pad(arr, (0, 64 - arr.size), 'constant')

    # On mélange le tableau
    np.random.shuffle(arr)

    return arr


import numpy as np
import torch
from torch.utils.data import Dataset


class RandomArrayDataset(Dataset):
    def __init__(self, num_samples):
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Utilisez la fonction create_random_array pour générer l'entrée et la sortie
        x = create_random_array()
        y = x.copy()  # les sorties sont identiques aux entrées

        # Convertissez les tableaux numpy en tenseurs PyTorch
        return torch.from_numpy(x).float(), torch.from_numpy(y).float()


class NN(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.function_1 = nn.Linear(input_size, 64)
        self.function_2 = nn.Linear(64, 32)
        self.function_3 = nn.Linear(32, 64)
        self.function_4 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.function_1(x)
        x = functional.relu(x)

        x = self.function_2(x)
        x = functional.relu(x)

        x = self.function_3(x)
        x = functional.relu(x)

        x = self.function_4(x)
        x = functional.sigmoid(x)

        return x


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
print(torch.__version__, torch.version.cuda)

# Additional Info when using cuda
if device.type == 'cuda':
    print(torch.cuda.get_device_name(0))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_reserved(0) / 1024 ** 3, 1), 'GB')
    print(f'Max: {round(torch.cuda.max_memory_reserved() / 1024 ** 3, 1)}')
print(device)

input_size = 64
num_classes = 64

lr = 5e-4
batch_size = 64
num_epochs = 200

from torch.utils.data import DataLoader

# Instanciation du dataset et Création du DataLoader
train_dataset = RandomArrayDataset(num_samples=100_000)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = RandomArrayDataset(num_samples=200_000)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

model = NN(input_size, num_classes).to(device=device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)


def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)

            # Convertir les cibles de one-hot à des indices de classe
            y = torch.argmax(y, dim=1)
            y = y.to(device=device)
            x = x.reshape(x.shape[0], -1)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

        calcul = float(num_correct) / float(num_samples) * 100

        #print(f"Got {num_correct} / {num_samples} with accuracy {calcul}")

        model.train()
        return calcul


from tqdm import tqdm

best_accuracy = 0
for epoch in range(num_epochs):
    pbar = tqdm(enumerate(train_loader), total=len(train_loader), leave=False)
    for batch_idx, (data, targets) in pbar:
        data = data.to(device=device)

        # Convertir les cibles de one-hot à des indices de classe
        targets = torch.argmax(targets, dim=1)
        targets = targets.to(device=device)

        scores = model.forward(data)

        _, predictions = scores.max(1)
        accuracy = (predictions == targets).sum()
        accuracy = f"{(float(accuracy) / batch_size) * 100:.6f}"

        loss = criterion(scores, targets)

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

        pbar.set_description(f"epochs: {epoch + 1}/{num_epochs}  -  accuracy : {accuracy}  -  loss: {loss.item():.4f}")

    # At the end of each epoch, calculate accuracy on test data
    train_accuracy = check_accuracy(train_loader, model)
    test_accuracy = check_accuracy(test_loader, model)

    pbar.refresh()

    # Now print the test accuracy along with the training loss and accuracy
    print(f"\nEpoch: {epoch + 1}/{num_epochs} - Training Loss: {loss.item():.4f} - Training Accuracy: {train_accuracy:.4f} - Test Accuracy: {test_accuracy:.4f}")

    if best_accuracy < float(test_accuracy):
        best_accuracy = test_accuracy
        print("Ce modèle est meilleur que le précedent, sauvegarde en cours..")
        torch.save(model.state_dict(), 'auto_encoder.pth')

check_accuracy(train_loader, model)
check_accuracy(test_loader, model)
