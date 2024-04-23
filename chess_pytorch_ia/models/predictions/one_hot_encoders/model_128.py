"""
Hérite de la classe ChessModel et implémente un modèle de prédiction
de coups (sous forme de vecteur one-hot) à jouer sur une position d'échecs donnée
"""
from typing import Iterable

from torch import nn

from chess_pytorch_ia.models.prediction import PredictionModelChess


class PMC_OneHotEncoder_128(PredictionModelChess):
    def __init__(self,
                 input_size=768,
                 hidden_layers=(512, 256),
                 output_size=128,

                 hidden_activation=nn.ReLU(),
                 output_activation=nn.Sigmoid(),
                 ):

        super().__init__(input_size, hidden_layers, output_size)

        # On crée les couches cachées
        self.layers = nn.ModuleList()

        last_input_size = input_size

        for hidden_size in hidden_layers:
            self.layers.append(nn.Linear(last_input_size, hidden_size))
            self.layers.append(hidden_activation)

            last_input_size = hidden_size

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        x = self.output_activation(x)

        return x
