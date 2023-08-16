from pathlib import Path

import numpy
import torch
import matplotlib.pyplot as plt
from create_output import create_64_output_matrix
from models.auto_encoder import AutoEncoder

if __file__ == '__main__':
    ### Changer par le fichier de configuration
    latent_size = 16
    input_size = 64

    path = Path(__file__).parent.parent / 'train' / 'weights' / f'auto_encoder_{input_size}.pth'

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    # On charge le même modèle et le poid du meilleur modèle
    model = AutoEncoder(input_size, latent_size).to(device=device)
    weight = torch.load(f'{path}')
    model.load_state_dict(weight)
    model.eval()

    dataset = [create_64_output_matrix(idx) for idx in range(64)]
    dataset = torch.from_numpy(numpy.array(dataset)).float()
    dataset = dataset.to(device=device)

    model.forward(dataset)

    min_ = model.latent_space.min()
    max_ = model.latent_space.max()

    # Afficher le tenseur en utilisant une échelle de couleurs allant du min au max
    plt.imshow(model.latent_space.cpu().detach().numpy(), cmap='seismic', vmin=min_, vmax=max_)
    plt.colorbar()
    plt.show()
