import torch
from torch.utils.data import Dataset
import numpy


def create_128_data_matrix(idx: int):
    """
    Renvoie une matrice de 128 contenant deux 1,
    l'un sur les premiers 64 cellules (quelle pièce ce déplace)
    et le deuxième sur les 64 dernières (ou est-ce que cette pièce ce déplace)
    """
    matrix = numpy.zeros(128)

    index_first = (idx // 64) % 64  # modulo 64 pour qu'il puisse supporter les idx de +4096
    index_second = idx % 64 + 64

    matrix[index_first] = 1
    matrix[index_second] = 1

    return matrix


class RandomMoveDataset128(Dataset):
    def __init__(self, num_samples):
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Utilisez la fonction create_random_array pour générer l'entrée et la sortie
        x = create_128_data_matrix(idx)
        y = x.copy()  # les sorties sont identiques aux entrées

        # Convertissez les tableaux numpy en tenseurs PyTorch
        return torch.from_numpy(x).float(), torch.from_numpy(y).float()
