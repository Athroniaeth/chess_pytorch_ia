import pandas

from preprocess.preprocess_data import preprocess_data_explode_by_shift

dataframe = pandas.DataFrame({
    'FEN': ['patrick', 'potron'],
    'Moves': ['jean, marechal, sebastien', 'pierre, jeanne, benoit']
})


def test_explode_by_shift():
    preprocess_dataframe = preprocess_data_explode_by_shift(dataframe)

    excepted_dataframe = pandas.DataFrame({
        'FEN': ['patrick', 'jean', 'marechal', 'potron', 'pierre', 'jeanne'],
        'Moves': ['jean', 'marechal', 'sebastien', 'pierre', 'jeanne', 'benoit']
    })

    assert excepted_dataframe.equals(preprocess_dataframe), f"La fonction de preprocess de la données, 'explode_by_shift' n'a pas le comportement désiré."
