import pandas


def clean_data(dataframe: pandas.DataFrame):
    """
    Cette fonction accepte un DataFrame pandas et une liste de colonnes à conserver.
    Elle retourne un nouveau DataFrame qui ne contient que les colonnes spécifiées, converties en chaînes de caractères.
    """
    columns_to_keep = ['FEN', 'Moves']

    # Vérifier que toutes les colonnes à conserver sont présentes dans le DataFrame d'origine
    for column in columns_to_keep:
        if column not in dataframe.columns.to_list():
            raise ValueError(f"La colonne '{column}' n'est pas dans le DataFrame.")

    # Créer un nouveau DataFrame avec seulement les colonnes à conserver
    dataframe_clean = clean_data_filter_columns(dataframe, columns_to_keep)

    # S'assurer que toutes les colonnes soit en types de données 'str'
    dataframe_clean = clean_data_types_cast(dataframe_clean, str)

    # Garder seulement les colonnes nécessaires et supprimer les lignes avec des valeurs nulles
    dataframe_clean = clean_data_drop_empty_lines(dataframe_clean)

    return dataframe_clean


def clean_data_filter_columns(dataframe: pandas.DataFrame, columns_to_keep):
    return dataframe[columns_to_keep]


def clean_data_types_cast(dataframe: pandas.DataFrame, dtypes):
    for column in dataframe.columns:
        dataframe[column] = dataframe[column].astype(dtypes)
    return dataframe


def clean_data_drop_empty_lines(dataframe: pandas.DataFrame):
    return dataframe.dropna()
