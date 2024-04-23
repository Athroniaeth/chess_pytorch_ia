import abc
import itertools
from abc import abstractmethod
from typing import List

import chess
import numpy


class Tokenizer():
    def __init__(self):
        pass

    @staticmethod
    def fen_to_board(fen: str) -> chess.Board:
        """
        Convert a fen to a chess board.

        Args:
            fen: The fen to convert.
        Returns:
            The board representation of the fen.
        """
        return chess.Board(fen)

    @staticmethod
    def get_boards_difference(fen_board: str, fen_move: str) -> numpy.ndarray:
        """
        À partir de 2 échiquiers, renvoie la différence entre les deux sous forme de matrice de vérité.

        Cette fonction est utilisé pour obtenir à partir d'une position et d'un coup, la matrice du coup.
        Une matrice de 64x pour la position de base de la pièce et une matrice de 64x pour le mouvement de la pièce.
        """
        if fen_board == fen_move:
            raise Exception(f"Les 2 codes FEN doivent être différent '{fen_board}' == '{fen_move}'")

        board_1 = chess.Board(fen_board)
        board_2 = chess.Board(fen_move)

        # On inverse la liste pour commencer par le Roi
        # (si le rock est joué, on veut que ce soit le roi qui bouge et non la tour)
        generator_pieces = itertools.product(chess.PIECE_TYPES[::-1], chess.COLORS)

        for piece_type, piece_color in generator_pieces:
            matrice_1: chess.SquareSet = board_1.pieces(piece_type, piece_color)
            matrice_2: chess.SquareSet = board_2.pieces(piece_type, piece_color)

            # tolist() inverse la liste comme la fonction .mirror(), nous devons donc appeler .mirror() pour corriger cela
            difference_squareset: chess.SquareSet = matrice_1.difference(matrice_2)
            difference: List[bool] = difference_squareset.mirror().tolist()

            # Si la matrice de vérité contient la moindre difference, on retourne le résultat
            if any(difference):
                return numpy.array(difference)

        raise Exception(f"La fonction n'a pas pu trouver la moindre différences entres les 2 codes fens,\n{board_1}\n{board_2}")

    def board_to_tensor(self, board: chess.Board) -> numpy.ndarray:
        """
        Convert a chess board to a tensor.

        Args:
            board: The board to convert.
        Returns:
            The tensor representation of the board.
        """

    def tensor_to_board(self, tensor: numpy.ndarray) -> chess.Board:
        """
        Convert a tensor to a chess board. Useless for moment.

        Args:
            tensor: The tensor to convert.
        Returns:
            The board representation of the tensor.
        """

    def move_to_tensor(self, board: chess.Board, move: chess.Move) -> numpy.ndarray:
        """
        Convert a chess move to a tensor.

        Args:
            board: The board on which the move is played.
            move: The move to convert.

        Returns:
            The tensor representation of the move.
        """

    @abstractmethod
    def tensor_to_move(self, tensor: numpy.ndarray) -> chess.Move:
        """
        Convert a tensor to a chess move.

        Args:
            tensor: The tensor to convert.
        Returns:
            The move representation of the tensor.
        """


token = Tokenizer()
print(token.get_boards_difference('r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3',
                                 'r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3').reshape((8, 8)))
print(token.get_boards_difference('r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3',
                                 'r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3').reshape((8, 8)))
