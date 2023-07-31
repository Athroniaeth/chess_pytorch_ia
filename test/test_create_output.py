from create_output import create_64_output_matrix
from create_output import create_128_output_matrix


def test_create_64_output_matrix():
    list_matrix = []

    for idx in range(64):
        matrix = create_64_output_matrix(idx)
        list_matrix.append(matrix)

    # Il y'a 64 cases, mois 1 case prise pour la position ou le déplacement de la pièce
    # (cela dépend si la matrice sert pour la position initiale de la pièce ou son déplacement)
    number_of_zeros = 64 - 1

    assert all(matrix.tolist().count(0) == number_of_zeros for matrix in list_matrix), f"Il y'a un nombre incohérent de '0', nous devons avoir {number_of_zeros} zeros sur cette matrice."
    assert all(matrix.tolist().count(1) == 1 for matrix in list_matrix), f"Il y'a un nombre incohérent de '1', il doit avoir un seul '1' sur toutes les cases de la matrice."


def test_create_128_output_matrix():
    list_matrix = []

    for idx in range(64 ** 2):
        matrix = create_128_output_matrix(idx)
        list_matrix.append(matrix)

    # Il y'a 2x 64 cases, mois 2 cases pris pour le déplacement de la pièce
    number_of_zeros = 64 * 2 - 2

    ## Doublons avec la deuxième vérification
    ## assert all(matrix.tolist().count(1) == 2 for matrix in list_matrix)

    assert all(matrix.tolist().count(0) == number_of_zeros for matrix in list_matrix), f"Il y'a un nombre incohérent de '0', nous devons avoir {number_of_zeros} zeros pour chaque matrice."
    ### Séparer les 2 vérifications ?
    assert all(matrix.tolist()[:64].count(1) == 1 and matrix.tolist()[64:128].count(1) == 1 for matrix in list_matrix), f"Il y'a un nombre incohérent de '1', il doit avoir un seul 1 sur les 64 premières cases de la matrice, et un seul 1 sur les 64 suivantes."
