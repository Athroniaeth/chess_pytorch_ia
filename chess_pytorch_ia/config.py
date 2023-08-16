from pathlib import Path
from typing import Dict, Any
import toml


class Config:
    _content: str = ""

    def __init__(self):
        raise TypeError('Config est une classe statique et ne peux être instanciée.')

    @staticmethod
    def load() -> Dict[str, Any]:
        if not Config._content:
            path = Path(__file__).parent / 'config.toml'
            content = toml.load(f'{path}')
            Config.content = content
        return Config.content

config = Config.load()
print(config)
