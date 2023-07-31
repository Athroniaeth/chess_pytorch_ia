import itertools
from typing import List

import chess
import numpy
import numpy as np
import pandas


def preprocess_data(dataframe: pandas.DataFrame):
    """
    Cette fonction accepte un DataFrame pandas.
    Elle applique au dataframe une explosion par décalage, ce qui permet de passer d'un
    dataframe de 'position de base': 'suite de meilleurs coups' à 'position de base': 'meilleur position suivante'.
    """

    # Applique la fonction d'explosion par décalage au dataframe
    preprocess_dataframe = preprocess_data_explode_by_shift(dataframe)

    return preprocess_dataframe


def preprocess_data_explode_by_shift(dataframe: pandas.DataFrame) -> pandas.DataFrame:
    """
    Fonction qui permet d'exploser les colonnes, mais à la place d'avoir le premier élement
    pour chaque nouvelle ligne, nous avons le dernier élement de la deuxième colonne (on explose en décallant)

    :param dataframe: pandas.DataFrame
    :return preprocess_dataframe: pandas.DataFrame
    """

    # Divisez la chaîne dans la colonne 'Moves' pour convertir chaque chaîne en une liste
    dataframe['Moves'] = dataframe['Moves'].str.split(',')

    # Utiliser 'explode' pour transformer chaque élément de la liste en une nouvelle ligne
    dataframe = dataframe.explode('Moves')

    # Enlever les espaces en début et fin de chaque élément dans 'Moves'
    dataframe['Moves'] = dataframe['Moves'].str.strip()

    # On prend le premier FEN (position de base) de chaque puzzle (sans doublons)
    # "dataframe['FEN']" ne fonctionne pas car il ne ce base pas sur un regroupement
    first_values = dataframe.groupby('FEN')['FEN'].transform('first')

    # On remplace la colonne FEN (position de base) par celle de Moves, en décallant de 1 vers le bas,
    # Cela permet d'avoir en première colonne la dernière position du même puzzle, comme ça on peut
    # remplacer les cellules vides par le FEN de base du puzzle
    series_fen = dataframe.groupby('FEN')['Moves']
    series_fen = series_fen.shift(1)
    series_fen = series_fen.fillna(first_values)

    dataframe['FEN'] = series_fen

    return dataframe.reset_index(drop=True)


def difference_fen_array(fen_base: str, fen_move: str) -> np.ndarray:
    """
    Renvoie la première différence detecté entre 2 code FEN en comparant les échiquiers

    Si vous mettez d'abord le fen de base, puis le fen du coup suivant, vous devriez avoir une
    matrice de vérité indiquant l'emplacement de base de la pièce déplacé.

    Si vous mettez d'abord le fen du coup suivant, puis le fen de base, vous devriez avoir une
    matrice de vérité indiquant l'emplacement actuel de la pièce déplacé.

    :param fen_base: code FEN du premier échiquier servant de base pour la comparer au 2ème FEN
    :param fen_move: code FEN du premier échiquier servant de différenciateur pour calculer la différence
    :return: Renvoie un numpy.array contenant une matrice de vérité sur la différence entre les 2 échiquiers
    """
    if fen_base == fen_move:
        raise Exception(f"Les 2 codes FEN doivent être différent '{fen_base}' == '{fen_move}'")

    board_1 = chess.Board(fen_base)
    board_2 = chess.Board(fen_move)

    # 'chess.PIECE_TYPES[::-1]' permet de commencer par le Roi pour verifier le rock
    for piece_type, piece_color in itertools.product(chess.PIECE_TYPES[::-1], chess.COLORS):
        matrice_1: chess.SquareSet = board_1.pieces(piece_type, piece_color)
        matrice_2: chess.SquareSet = board_2.pieces(piece_type, piece_color)

        difference_squareset: chess.SquareSet = matrice_1.difference(matrice_2)
        # tolist() inverse la liste ,comme la fonction .mirror(), nous devons donc appeller .mirror() pour corriger cela
        difference: List[bool] = difference_squareset.mirror().tolist()

        # Si la matrice de vérité contiens la moindre difference, on retourne le résultat
        if any(difference):
            return numpy.array(difference)

    raise Exception(f"La fonction n'a pas pu trouver la moindre différences entres les 2 codes fens,\n{board_1}\n{board_2}")


print(difference_fen_array('r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3', 'r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3').reshape((8, 8)))
print(difference_fen_array('r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3', 'r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3').reshape((8, 8)))
