"""
Hérite de la classe ChessModel et implémente un modèle de prédiction
de coups (sous forme de vecteur one-hot) à jouer sur une position d'échecs donnée
"""
from typing import Iterable

from torch import nn

from chess_pytorch_ia.models.auto_encoder import MoveAutoEncoder


class MoveAutoEncoder128(MoveAutoEncoder):
    """Implementation of MoveAutoEncoder with 64 input size."""

    def __init__(self, input_size=128, hidden_layers=(32,), latent_size=8,
                 hidden_activation=nn.ReLU(), output_activation=nn.Softmax(dim=1), *args, **kwargs):

        super().__init__(input_size=input_size, hidden_layers=hidden_layers, latent_size=latent_size,
                         hidden_activation=hidden_activation, output_activation=output_activation, *args, **kwargs)

        latent_size = latent_size // 2
        # delete last layer and activation
        # split into 2 separate outputs to apply a softmax
        self.decoder = self.encoder[:-2]
        self.decoder_output_1 = nn.Linear(hidden_layers[-1], latent_size)
        self.decoder_output_2 = nn.Linear(hidden_layers[-1], latent_size)
        self.output_activation = output_activation

    def decode(self, x):
        """Decode the input data."""
        x = self.decoder(x)
        x1 = self.decoder_output_1(x)
        x2 = self.decoder_output_2(x)
        return self.output_activation(x1), self.output_activation(x2)
