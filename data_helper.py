import pandas as pd


def load_data (name, dataPath, index_col = None):
    return  pd.read_csv(dataPath / f'{name}.csv', na_values=['?'], 
                        index_col=index_col)

def make_categ (df, l_cols_categ):
    for col in l_cols_categ:
        df[col] = df[col].astype("category")
    return df