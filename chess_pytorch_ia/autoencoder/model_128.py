import numpy
from torch import nn


class AutoEncoder128(nn.Module):
    def __init__(
            self,
            input_size=128,
            hidden_size=64,
            latent_size=8,
            hidden_activation=nn.Identity,
            output_activation=nn.Sigmoid,
            *args, **kwargs
    ):
        super().__init__(*args, **kwargs)

        # Encodeur
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            hidden_activation(),
            nn.Linear(hidden_size, latent_size),
            hidden_activation()
        )

        # Décodeur
        self.decoder = nn.Sequential(
            nn.Linear(latent_size, hidden_size),
            hidden_activation(),
            nn.Linear(hidden_size, input_size),
            output_activation()
        )

    def forward(self, x):
        latent_space = self.encode(x)
        return self.decode(latent_space)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)
