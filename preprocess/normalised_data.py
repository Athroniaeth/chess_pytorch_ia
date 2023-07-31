import chess
import pandas


def normalised_data(dataframe: pandas.DataFrame):
    """Transforme le dataframe en dataframe lisible par le réseau de neurones"""

    # Transforme les codes FEN en chess.board()
    normalised_dataframe = normalised_data_fen_to_chessboard(dataframe)

    return normalised_dataframe


def normalised_data_fen_to_chessboard(dataframe: pandas.DataFrame):
    normalised_dataframe = pandas.DataFrame({'inputs': [], 'outputs': []})
    normalised_dataframe['inputs'] = dataframe['FEN'].apply(fen_to_board)  # Trouvée une fonction plus rapide
    normalised_dataframe['inputs'] = dataframe['Moves'].apply(fen_to_board)  # Trouvée une fonction plus rapide
    return normalised_dataframe


def fen_to_board(fen: str) -> chess.Board:
    chess.Board(fen)
