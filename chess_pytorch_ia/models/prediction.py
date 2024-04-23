""" Classe abstraite pour le modèle de prediction de coups à jouer aux échecs"""
import abc
from typing import Iterable

import numpy
from torch import nn


class PredictionModelChess(nn.Module, metaclass=abc.ABCMeta):
    """
    Classe abstraite pour le modèle de prediction de coups à jouer aux échecs
    """

    def __init__(self, input_size: int, hidden_layers: Iterable[int], output_size: int, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        raise NotImplementedError("La fonction forward doit être implémentée")


