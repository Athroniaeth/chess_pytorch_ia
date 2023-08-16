"""
Vous trouverez ici les fonctions globales qu'ont besoin
les scripts d'entrainement dans le dossier 'train'
"""

import torch
from torch import nn
from torch.utils.data import DataLoader


def check_accuracy(loader: DataLoader, model: nn.Module, device: torch.device):
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

        model.train()
        return calcul
