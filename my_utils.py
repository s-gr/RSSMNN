# import numpy as np
import pandas as pd

PATH = 'dataset/'                                                                                   # set path variable for dataset

table_names = ['train', 'store', 'store_states', 'state_names', 'googletrend', 'weather', 'test']   # input table names
tables = [pd.read_csv(f'{PATH}{fname}.csv', low_memory=False) for fname in table_names]             # create list of DataFrames

for t in tables:
    # print(t.head())                                                                                 # show first 5 lines of all data frames
    df = t.select_dtypes(include='number')
    df = df.head()
    # df = df.describe()
    df = df.mean(axis=1, )
    print(df)

    quit()
