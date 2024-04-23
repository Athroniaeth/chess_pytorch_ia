"""
Hérite de la classe ChessModel et implémente un modèle de prédiction
de coups (sous forme de vecteur one-hot) à jouer sur une position d'échecs donnée
"""
from typing import Iterable

from torch import nn

from chess_pytorch_ia.models.prediction import PredictionModelChess


class PMC_OneHotEncoder_64(PredictionModelChess):
    def __init__(self,
                 input_size=768,
                 hidden_layers=(512, 256),
                 output_size=128,

                 hidden_activation=nn.ReLU(),
                 output_activation=nn.Softmax(dim=1),
                 ):

        super().__init__(input_size, hidden_layers, output_size)

        # On crée les couches cachées
        self.layers = nn.ModuleList()

        last_input_size = input_size

        for hidden_size in hidden_layers:
            self.layers.append(nn.Linear(last_input_size, hidden_size))
            self.layers.append(hidden_activation)

            last_input_size = hidden_size

        # On crée la couche de sortie
        self.piece_to_move_head = output_activation(hidden_layers[-1], output_size)
        self.move_to_case_head = output_activation(hidden_layers[-1], output_size)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        piece_to_move = nn.Softmax(dim=1)(self.piece_to_move_head(x))
        move_to = nn.Softmax(dim=1)(self.move_to_case_head(x))

        return piece_to_move, move_to
