import pickle
import os, re
import random
from datetime import date, timedelta
import numpy as np
import pandas as pd
import pandas_summary as psm
from sklearn import preprocessing

PATH = 'dataset/'  # set path variable for dataset
table_names = ['train', 'store', 'store_states', 'test']


def load_data(path, table_names):
    tables = []                                                                                 # create empty list
    tables = [pd.read_csv(f'{path}{fname}.csv', low_memory=False) for fname in table_names]     # create list of DataFrames
    for table in tables:
        if 'Date' in table.columns:
            table['Date'] = pd.to_datetime(table['Date'])                                       # Change dtype of Date to datetime
        if 'State' in table.columns:
            table.loc[table.State == 'HB,NI', 'State'] = 'NI'                                   # Change 'HB,NI' to 'NI'
        if 'Promo2SinceWeek' in table.columns:
            table['Promo2SinceWeek'].fillna(0, inplace=True)
        if 'Promo2SinceYear' in table.columns:
            table['Promo2SinceYear'].fillna(0, inplace=True)
        if 'CompetitionOpenSinceMonth' in table.columns:
            table['CompetitionOpenSinceMonth'].fillna(0, inplace=True)
        if 'CompetitionOpenSinceYear' in table.columns:
            table['CompetitionOpenSinceYear'].fillna(0, inplace=True)
        if 'PromoInterval' in table.columns:
            table['PromoInterval'].fillna(0, inplace=True)
        if 'CompetitionDistance' in table.columns:
            table['CompetitionDistance'].fillna(0, inplace=True)
        print(table.head())
    return tables


def generate_features(joined_df):
    joined_df['Year'] = joined_df.Date.dt.year
    joined_df['Month'] = joined_df.Date.dt.month
    joined_df['Day'] = joined_df.Date.dt.day


def join_dfs(left, right, left_on, right_on = None, suffix = '_y'):                             # left-join two data frames
    if right_on is None:
        right_on = left_on
    return left.merge(right, how='left', left_on=left_on, right_on=right_on, suffixes=("", suffix))


train, store, store_states, test = load_data(PATH, table_names)                                 # Load tables to data frames

train.sort_values(['Date', 'Store'], ascending = True, inplace=True)
print(train.head().T)
joined = join_dfs(train, store, 'Store')                                                        # Join train and store set at Store column
joined = join_dfs(joined, store_states, 'Store')                                                # Join joined and store_states at Store column

generate_features(joined)


print(joined.head().T)
print(psm.DataFrameSummary(joined).summary().T)
# print(len(joined.loc[joined.Sales == 0, 'Open'].index))
# print(joined[['DayOfWeek', 'Customers']].head().T)
joined = joined.loc[(joined['Sales'] != 0) & (joined['Open'] != 0)]
train_data_X = joined[['Open', 'Store', 'DayOfWeek', 'Promo', 'Year', 'Month', 'Day', 'State']]
train_data_y = joined['Sales']
print(train_data_X.head().T)
print(train_data_y.head().T)

print('Number of train datapoints: ', len(train_data_y))

print('Minimum Value: ', min(train_data_y))
print('Maximum Value: ', max(train_data_y))

train_data_X = train_data_X.values
full_X = train_data_X
les = []
train_data_y = train_data_y.values

for i in range(train_data_X.shape[1]):
    le = preprocessing.LabelEncoder()
    le.fit(full_X[:, i])
    les.append(le)
    train_data_X[:, i] = le.transform(train_data_X[:, i])

with open('tmp/les.pickle', 'wb') as f:
    pickle.dump(les, f, -1)

print(train_data_y[0])
print(train_data_y.shape)

with open('tmp/feature_train_data.pickle', 'wb') as f:
    pickle.dump((train_data_X, train_data_y), f, -1)
    print(train_data_X[0], train_data_y[0])
