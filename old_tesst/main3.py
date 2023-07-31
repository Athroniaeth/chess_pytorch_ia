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
    # On crée un array contenant deux 1
    arr = np.array([1, 1])

    # On ajoute les zéros pour atteindre une taille de 128
    arr = np.pad(arr, (0, 128 - arr.size), 'constant')

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

class VAE(nn.Module):
    def __init__(self, input_size, hidden_size, latent_size):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2_1 = nn.Linear(hidden_size, latent_size)
        self.fc2_2 = nn.Linear(hidden_size, latent_size)
        self.fc3 = nn.Linear(latent_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, input_size)

    def encode(self, x):
        x = self.fc1(x)
        h2 = functional.relu(x)
        return self.fc2_1(h2), self.fc2_2(h2)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        z = self.fc3(z)
        h3 = functional.relu(z)
        h3 = self.fc4(h3)
        return torch.sigmoid(h3)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

def loss_function(recon_x, x, mu, logvar):
    BCE = functional.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

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

        calcul = f"{float(num_correct) / float(num_samples) * 100:.2f}"

        print(f"Got {num_correct} / {num_samples} with accuracy {calcul}")

        model.train()
        return calcul


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

input_size = 128
hidden_size = 64
latent_size = 32

lr = 1e-3
batch_size = 32
num_epochs = 100

from torch.utils.data import DataLoader

# Instanciation du dataset et Création du DataLoader
train_dataset = RandomArrayDataset(num_samples=100_000)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = RandomArrayDataset(num_samples=100_000)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)


model = VAE(input_size, hidden_size, latent_size).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

model.train()
def accuracy(output, target):
    """Compute the accuracy for a batch.

    Args:
    output (torch.Tensor): The output from the model, of shape [batch_size, num_classes]
    target (torch.Tensor): The ground truth labels, of shape [batch_size, num_classes]

    Returns:
    float: The accuracy of the model on this batch
    """
    _, predicted = torch.max(output, 1)
    target_indices = torch.argmax(target, 1)
    correct = (predicted == target_indices).float()  # convert into float for division
    acc = correct.sum() / len(correct)
    return acc



for epoch in range(num_epochs):
    train_loss = 0
    train_accuracy = 0
    for batch_idx, (data, targets) in enumerate(train_loader):
        data = data.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()

        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        acc = accuracy(recon_batch, targets)

        loss.backward()
        train_loss += loss.item()
        train_accuracy += acc.item()
        optimizer.step()

    percentage = train_loss / len(train_loader.dataset)
    average_accuracy = train_accuracy / len(train_loader)
    print(f'Epoch: {epoch} Average loss: {percentage:.4f}, Average accuracy: {average_accuracy:.4f}')







check_accuracy(train_loader, model)
check_accuracy(test_loader, model)
