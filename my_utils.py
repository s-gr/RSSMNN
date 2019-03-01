import os, re
from datetime import date, timedelta
import numpy as np
import pandas as pd
import pandas_summary as psm

PATH = 'dataset/'  # set path variable for dataset

table_names = ['train', 'store', 'store_states', 'state_names', 'googletrend', 'weather', 'test']  # input table names
tables = [pd.read_csv(f'{PATH}{fname}.csv', low_memory=False) for fname in table_names]  # create list of DataFrames

# for t in tables:
#     print(t.head(3))

train, store, store_states, state_names, googletrend, weather, test = tables


def join_df(left, right, left_on, right_on=None, suffix='_y'):
    if right_on is None: right_on = left_on
    return left.merge(right, how='left', left_on=left_on, right_on=right_on, suffixes=('', suffix))


def get_weekday(year, week):
    d = date(year, 1, 1)
    if d.weekday() > 3:
        d = d + timedelta(7 - d.weekday())
    else:
        d = d - timedelta(d.weekday())
    dlt = timedelta(days=(week - 1) * 7)
    return d + dlt #, d + dlt +


new_df = join_df(weather, state_names, 'file', 'StateName')

print(new_df.head())

joined_test = pd.read_pickle('dataset/joined_test')

for i in list(joined_test):
    print(i)

print(joined_test['Promo2'].head())
joined_test.head().to_csv('test.csv')

joined_test['Promo2Since'] = pd.to_datetime(
    joined_test.apply(lambda x:
                      get_weekday(
                          x.Promo2SinceYear, x.Promo2SinceWeek
                      ), axis=1
                      )
)

joined_test['Promo2Days'] = joined_test.Date.subtract(joined_test['Promo2Since']).dt.days

print(joined_test['Promo2Days'].head())

# for y,w in zip(joined_test['Promo2SinceYear'], joined_test['Promo2SinceWeek']):
#     print(y)
#     print(w)
#
# d_test = date(12,1,1)
#
