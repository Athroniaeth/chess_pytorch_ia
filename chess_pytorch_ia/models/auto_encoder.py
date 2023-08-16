"""
Ceci est un AutoEncoder de test pour voir si un espace lattent de 16x (8+8) à la place d'une sortie de 128x (64+64)
est plus efficace ou non lors de l'entrainement du modèle principale, ici on s'en fiche d'avoir un surentrainement
car on veut précisement que l'auto encoder arrive a nous sortir exactement la même sortie.

------------

Ceci est un AutoEncoder de test pour voir si un espace lattent de 16x (8+8) à la place d'une sortie de 128x (64+64)
est plus efficace ou non lors de l'entrainement du modèle principale, ici on s'en fiche d'avoir un surentrainement
car on veut précisement que l'auto encoder arrive a nous sortir exactement la même sortie.

Remarque : On ne crée en donnée d'entrainement que une matrice de 64, car on utilisera 2x ce modèle pour la sortie du
modèle principale, une fois pour la position de la pièce a déplacer, une dernière fois pour sa position finale
"""

import numpy
from torch import nn


class AutoEncoder(nn.Module):
    latent_space: numpy.ndarray or None

    def __init__(self, input_size=64, latent_size=8):
        super(AutoEncoder, self).__init__()

        # Encodeur
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ReLU(),
            nn.Linear(32, latent_size),
            nn.ReLU()
        )

        # Decodeur
        self.decoder = nn.Sequential(
            nn.Linear(latent_size, 32),
            nn.ReLU(),
            nn.Linear(32, input_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        self.latent_space = self.encoder(x)
        x = self.decoder(self.latent_space)
        return x

    def encode(self, x):
        self.latent_space = self.encoder(x)
        return self.latent_space

    def decode(self, x):
        return self.decoder(x)
# La commande (courte) pylint pour faire le HTML :
# py