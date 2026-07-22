import pandas as pd


def build_transition_matrix(series):
    return pd.crosstab(
        series.shift(),
        series,
        normalize="index"
    )


def predict_next(state, matrix):

    if state not in matrix.index:
        return state

    # Chagua state yenye probability kubwa zaidi
    return matrix.loc[state].idxmax()