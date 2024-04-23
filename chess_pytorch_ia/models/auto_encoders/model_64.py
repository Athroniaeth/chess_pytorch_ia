"""
Hérite de la classe ChessModel et implémente un modèle de prédiction
de coups (sous forme de vecteur one-hot) à jouer sur une position d'échecs donnée
"""
from typing import Iterable

from torch import nn

from chess_pytorch_ia.models.auto_encoder import MoveAutoEncoder


class MoveAutoEncoder64(MoveAutoEncoder):
    """Implementation of MoveAutoEncoder with 64 input size."""

    def __init__(self, *args, **kwargs):
        super().__init__(
            input_size=64,
            hidden_layers=[32],
            latent_size=8,
            hidden_activation=nn.ReLU(),
            output_activation=nn.Softmax(dim=1),
            *args,
            **kwargs
            )
