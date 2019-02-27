# import numpy as np
import pandas as pd
import pandas_summary as psm

PATH = 'dataset/'                                                                                   # set path variable for dataset

table_names = ['train', 'store', 'store_states', 'state_names', 'googletrend', 'weather', 'test']   # input table names
tables = [pd.read_csv(f'{PATH}{fname}.csv', low_memory=False) for fname in table_names]             # create list of DataFrames

for t in tables:
    print(psm.DataFrameSummary(t).summary())                                                        # Display overview of DataFrames using pandas_summary object


train, store, store_states, state_names, googletrend, weather, test = tables                        #

print(str(len(train)) + ', ' + str(len(test)))

