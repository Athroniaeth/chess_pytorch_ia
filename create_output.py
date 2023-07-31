import numpy


def create_64_output_matrix(idx: int):
    """
    Renvoie une matrice de 64 contenant un seul 1, ce output est prévu pour être utilisé deux fois
    la première utilisation sera pour savoir quelle pièce ce déplace
    la deuxième utilisation sera pour savoir ou est-ce que cette pièce ce déplace
    """
    matrix = [0 for _ in range(64)]
    matrix[idx % 64] = 1  # modulo 64 pour qu'il puisse supporter les idx de +64
    arr = numpy.array(matrix)

    return arr


def create_128_output_matrix(idx: int):
    """
    Renvoie une matrice de 128 contenant deux 1,
    l'un sur les premiers 64 cellules (quelle pièce ce déplace)
    et le deuxième sur les 64 dernières (ou est-ce que cette pièce ce déplace)
    """
    matrix = numpy.zeros(128)

    index_first = (idx // 64) % 64  # modulo 64 pour qu'il puisse supporter les idx de +4096
    index_second = idx % 64 + 64

    matrix[index_first] = 1
    matrix[index_second] = 1

    return matrix
