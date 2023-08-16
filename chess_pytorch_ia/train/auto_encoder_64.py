import torch
from pathlib import Path

from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from create_output import create_64_output_matrix
from models.auto_encoder import AutoEncoder
from train import check_accuracy


class RandomArrayDataset(Dataset):
    def __init__(self, num_samples):
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Utilisez la fonction create_random_array pour générer l'entrée et la sortie
        x = create_64_output_matrix(idx)
        y = x.copy()  # les sorties sont identiques aux entrées

        # Convertissez les tableaux numpy en tenseurs PyTorch
        return torch.from_numpy(x).float(), torch.from_numpy(y).float()

if __name__ == '__main__':
    latent_size = 16
    input_size = 64

    lr = 5e-3
    batch_size = 100
    num_epochs = 200

    path = Path(__file__).parent / 'weights' / f'auto_encoder_{input_size}.pth'
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    # Instanciation du dataset et Création du DataLoader
    dataset = RandomArrayDataset(num_samples=100_000)
    loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

    model = AutoEncoder(input_size, latent_size).to(device=device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_accuracy = 0
    for epoch in range(num_epochs):
        pbar = tqdm(enumerate(loader), total=len(loader), leave=False)
        for batch_idx, (data, targets) in pbar:
            data = data.to(device=device)

            # Convertir les cibles de one-hot à des indices de classe
            targets = torch.argmax(targets, dim=1)
            targets = targets.to(device=device)

            scores = model.forward(data)

            _, predictions = scores.max(1)
            accuracy = (predictions == targets).sum()
            accuracy = f"{(float(accuracy) / batch_size) * 100:.2f}"

            loss = criterion(scores, targets)

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

            pbar.set_description(f"epochs: {epoch + 1}/{num_epochs}  -  accuracy : {accuracy}  -  loss: {loss.item():.4f}")

        accuracy = check_accuracy(loader, model, device=device)

        pbar.refresh()

        # On affiche la précision sur le dataset de 'test' avec le coût d'entrainement.
        print(f"\nEpoch: {epoch + 1}/{num_epochs} - Training Loss: {loss.item():.4f} - Accuracy: {accuracy}")

        # On sauvegarde les poids seulement si le modèle est meilleur que le précedent
        if float(best_accuracy) < float(accuracy):
            best_accuracy = accuracy
            print("Ce modèle est meilleur que le précedent, sauvegarde en cours..")
            torch.save(model.state_dict(), f'{path}')

    check_accuracy(loader, model, device)
