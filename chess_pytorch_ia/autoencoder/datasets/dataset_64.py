import torch
from torch.utils.data import Dataset
import numpy


def create_64_data_matrix(idx: int):
    """
    Renvoie une matrice de 64 contenant un seul 1, ce output est prévu pour être utilisé deux fois
    la première utilisation sera pour savoir quelle pièce ce déplace
    la deuxième utilisation sera pour savoir ou est-ce que cette pièce ce déplace
    """
    matrix = [0 for _ in range(64)]
    matrix[idx % 64] = 1  # modulo 64 pour qu'il puisse supporter les idx de +64
    arr = numpy.array(matrix)

    return arr


class RandomMoveDataset64(Dataset):
    def __init__(self, num_samples=64):
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Utilisez la fonction create_random_array pour générer l'entrée et la sortie
        x = create_64_data_matrix(idx)
        y = x.copy()  # les sorties sont identiques aux entrées

        # Convertissez les tableaux numpy en tenseurs PyTorch
        return torch.from_numpy(x).float(), torch.from_numpy(y).float()
