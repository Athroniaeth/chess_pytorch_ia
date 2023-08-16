import chess
import numpy


class ParserFEN:

    def fen_to_numpy(self, board: chess.Board):
        """preprocess_data to inputs NN (inputs datasets)"""
        ...

    def numpy_to_fen(self, array: numpy.ndarray):
        """inputs NN to preprocess_data (inputs)"""
        ...

    # # # # # # # # # # # # # # # # # # # #

    def move_to_numpy(self, board: chess.Board, move):
        """preprocess_data to inputs NN (inputs datasets)"""
        ...

    def numpy_to_move(self, board: chess.Board, move):
        """inputs NN to preprocess_data (inputs)"""
        ...
