import pandas

from preprocess.clean_data import clean_data_filter_columns, clean_data_drop_empty_lines, clean_data, clean_data_types_cast

# Préparer les données pour les tests
dataframe = pandas.DataFrame({
    'FEN': ['patrick', 'potron'],
    'Moves': ['jean, marechal, sebastien', 'pierre, jeanne, benoit']
})
columns_to_keep = ['FEN', 'Moves']


def test_clean_data_filter_columns():
    dataframe_filtered = clean_data_filter_columns(dataframe, columns_to_keep)
    assert set(dataframe_filtered.columns) == set(columns_to_keep)


def test_clean_data_types_cast():
    dataframe_casted = clean_data_types_cast(dataframe[columns_to_keep], str)
    assert all(dataframe_casted.dtypes == object)


def test_clean_data_drop_empty_lines():
    dataframe_dropped = clean_data_drop_empty_lines(dataframe[columns_to_keep])
    assert dataframe_dropped.isnull().sum().sum() == 0


def test_clean_data():
    dataframe_cleaned = clean_data(dataframe)
    assert set(dataframe_cleaned.columns) == set(columns_to_keep)
    assert any(dataframe_cleaned.dtypes == object)
    assert dataframe_cleaned.isnull().sum().sum() == 0
