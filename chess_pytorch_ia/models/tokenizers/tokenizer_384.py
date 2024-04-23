import chess
import numpy

from chess_pytorch_ia.models.tokenizer import Tokenizer


class Tokenizer368(Tokenizer):
    """Implementation of Tokenizer with 368 input size."""

    def __init__(self):
        super().__init__()

    def board_to_tensor(self, board):
        pass

    def tensor_to_board(self, tensor):
        pass

    def move_to_tensor(self, board, move):
        pass

    def tensor_to_move(self, tensor):
        pass
