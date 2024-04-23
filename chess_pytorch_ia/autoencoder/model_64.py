import numpy
from torch import nn


class AutoEncoder64(nn.Module):
    def __init__(
            self,
            input_size=64,
            hidden_size=32,
            latent_size=4,
            *args, **kwargs
    ):
        super().__init__(*args, **kwargs)

        # Encodeur
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, latent_size),
            nn.ReLU(),
        )

        # Décodeur
        self.decoder = nn.Sequential(
            nn.Linear(latent_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, input_size),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        latent_space = self.encode(x)
        return self.decode(latent_space)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)
