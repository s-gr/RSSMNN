import warnings
import os
import re
from datetime import date, timedelta
import numpy as np
import pandas as pd
import pandas_summary as psm
from pandas.api.types import is_string_dtype, is_numeric_dtype
import sklearn
from sklearn.preprocessing import LabelEncoder, Imputer, StandardScaler
import matplotlib.pyplot as plt


PATH_data = 'dataset/'                                                                              # setting PATH to data

# Loading the tables

table_names = ['train', 'store', 'store_states']                             # table names to import
tables = [pd.read_csv(f'{PATH_data}{fname}.csv', low_memory=False) for fname in table_names]        # create list of data frames

for table in tables:
    if 'Date' in table.columns:
        table['Date'] = pd.to_datetime(table['Date'])
    print(table.head())
    print(psm.DataFrameSummary(table).summary())

    if 'Open' in table.columns:
        plt.figure()
        table['Open'].value_counts(normalize=True).plot(kind='bar', rot=0, grid=True, alpha=0.5, width=1)
        plt.title('Open')
        plt.ylabel('Frequency' + ' (normalized to ' + str(len(table['Open'].index)) + ')')
        plt.xlabel('Uniques')
        # plt.savefig('im1.png')
        plt.show()

    if 'StateHoliday' in table.columns:
        plt.figure()
        table['StateHoliday'].value_counts(normalize=True).plot(kind='bar', rot=0, grid=True, alpha=0.5, width=1)
        plt.title('StateHoliday')
        plt.ylabel('Frequency' + ' (normalized to ' + str(len(table['StateHoliday'].index)) + ')')
        plt.xlabel('Uniques')
        # plt.savefig('im2.png')
        plt.show()
    _ = input()
    print('\n')

# Hieraus eine Funkton schreiben, die listen von Columns und optionen bekommt. mach daraus dann einen subplot in einer Reihe