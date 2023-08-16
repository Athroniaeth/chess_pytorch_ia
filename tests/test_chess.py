# Fichier de test pour s'assurer que les fonctions utilisé de la librairie python-chess
# sont toujours les mêmes que celle utilisé pour ce projet
from typing import List

import chess


def test_chess_difference():
    """
    Test si la fonction 'difference' de l'objet chess.SquareSet fonctionne comme lors de
    la création du projet (Attention, la fonction tolist() inverse côté blanc/noir)
    :return:
    """
    excepted_output = [
        False, False, False, False, False, False, False, False,
        True,  True,  True,  True,  True,  True,  True,  True,
        False, False, False, False, False, False, False, False,
        False, False, False, False, False, False, False, False,
        False, False, False, False, False, False, False, False,
        False, False, False, False, False, False, False, False,
        False, False, False, False, False, False, False, False,
        False, False, False, False, False, False, False, False
    ]

    board = chess.Board()

    white_pawn: chess.SquareSet = board.pieces(chess.PAWN, chess.WHITE)
    black_pawn: chess.SquareSet = board.pieces(chess.PAWN, chess.BLACK)

    difference_squareset: chess.SquareSet = white_pawn.difference(black_pawn)
    difference: List[bool] = difference_squareset.tolist()

    assert difference == excepted_output, "La fonction de python-chess '<class 'chess.SquareSet'>'.difference() n'est plus identique à celle faite lors du projet"
