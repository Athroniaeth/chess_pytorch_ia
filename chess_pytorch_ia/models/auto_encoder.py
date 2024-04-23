""" Classe abstraite pour le modèle d'auto encodeur de coups à jouer aux échecs"""
import abc
from typing import Iterable, List

import numpy

from torch import nn


class MoveAutoEncoder(nn.Module, metaclass=abc.ABCMeta):
    """Abstract class for chess move autoencoder model.

    This class serves as the base for chess move prediction models by providing
    its latent space as output data (encode).
    """

    def __init__(self,
                 input_size: int,
                 latent_size: int,
                 hidden_layers: List[int],
                 hidden_activation: nn.Module,
                 output_activation: nn.Module,
                 *args,
                 **kwargs):
        """Initialize the MoveAutoEncoder.

        Args:
            input_size: The size of the input layer.
            hidden_layers: The sizes of the hidden layers.
            latent_size: The size of the latent layer.
            hidden_activation: Activation function for hidden layers.
            output_activation: Activation function for output layer.
        """
        super().__init__(*args, **kwargs)

        if hidden_layers is None:
            raise ValueError("hidden_layers must be defined.")

        self.encoder = self._build_hidden_layers(input_size, hidden_layers, hidden_activation, name='encoder')
        self.decoder = self._build_hidden_layers(latent_size, hidden_layers[::-1], hidden_activation, name='decoder')

        # Ajout de la couche de sortie
        self.encoder.add_module('encoder_output_layer', nn.Linear(hidden_layers[-1], latent_size))
        self.encoder.add_module('encoder_output_activation', output_activation)

        self.decoder.add_module('decoder_output_layer', nn.Linear(hidden_layers[0], latent_size))
        self.decoder.add_module('decoder_output_activation', output_activation)

    @staticmethod
    def _build_hidden_layers(input_size: int, layer_sizes: Iterable[int], activation: nn.Module, name: str) -> nn.Sequential:
        """Build a sequence of layers."""
        layers = nn.Sequential()

        for index, size in enumerate(layer_sizes):
            layer = nn.Linear(input_size, size)
            layers.add_module(f'{name}_hidden_layer_{index}', layer)
            layers.add_module(f'{name}_hidden_activation_{index}', activation)
            input_size = size

        return layers

    def forward(self, x):
        """Forward pass through the autoencoder."""
        latent_space = self.encode(x)
        return self.decode(latent_space)

    def encode(self, x):
        """Encode the input data."""
        return self.encoder(x)

    def decode(self, x):
        """Decode the input data."""
        return self.decoder(x)
