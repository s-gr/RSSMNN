import os.path
import numpy as np
import pandas as pd
import pandas_summary as psm
from sklearn import preprocessing

PATH = 'dataset/'  # set path variable for dataset
table_names = ['station', 'trip']


def load_data(path, table_names):
    with open(f'{path}new_trip.csv', 'w') as new_file:                                                  # Parse to DataFrame
        with open(f'{path}trip.csv', 'r') as file:
            for i, line in enumerate(file):
                if i == 50793:
                    new_file.write(line.split('trip_id', 1)[0] + '\n')
                else:
                    new_file.write(line)

    table_names[1] = 'new_trip'
    tables = [pd.read_csv(f'{path}{fname}.csv', low_memory=False) for fname in table_names]

    tables[1].drop(tables[1].index[:50793], inplace=True)
    tables[1] = tables[1].reset_index(drop=True)
    # tables[1].shape
    station, trip = tables

    for column in station.columns:
        if 'date' in column:
            station[column] = station[column].fillna('1900/01/01').astype(str)
            station[column] = pd.to_datetime(station[column])

    trip.drop(columns=['from_station_name', 'to_station_name'], inplace=True)                           # Drop Station Names
    for column in trip.columns:
        if 'time' in column:
            trip[column] = trip[column].fillna('1900/01/01').astype(str)
            trip[column] = pd.to_datetime(trip[column])

    # changing datatype of birthyear to int64 and set NAs to 1900
    trip['birthyear'] = trip['birthyear'].fillna(1900).astype(np.int64)

    # filling NAs in gender to 'NA'
    trip['gender'] = trip['gender'].fillna('NA')
    return station, trip


def generate_features(df):                                                                              # Generate new features
    df['start_Date'] = df.starttime.dt.date
    # df['start_HourOfDay'] = df.starttime.dt.hour
    df['start_DayOfWeek'] = df.starttime.dt.dayofweek
    # df_temp = df.groupby(['from_station_id', 'start_Date', 'start_HourOfDay'])['trip_id'].count()
    df_temp = df.groupby(['from_station_id', 'start_Date', 'start_DayOfWeek'])['trip_id'].count()
    df_temp = df_temp.reset_index()
    df_temp.rename(columns={'trip_id': 'trip_count'}, inplace=True)
    # df_temp = df_temp.sort_values(['start_Date', 'start_HourOfDay', 'from_station_id'])
    df_temp = df_temp.sort_values(['start_Date', 'start_DayOfWeek', 'from_station_id'])
    df_temp = df_temp.reset_index(drop=True)

    # df = df[['from_station_id', 'start_Date', 'start_HourOfDay']].loc[df[['from_station_id', 'start_Date', 'start_HourOfDay']].drop_duplicates().index]
    df = df[['from_station_id', 'start_Date', 'start_DayOfWeek']].loc[df[['from_station_id', 'start_Date', 'start_DayOfWeek']].drop_duplicates().index]
    # df = df.sort_values(['start_Date', 'start_HourOfDay', 'from_station_id'])
    df = df.sort_values(['start_Date', 'start_DayOfWeek', 'from_station_id'])
    df = df.reset_index(drop=True)

    df = pd.concat([df, df_temp], axis=1, sort=False)
    df = df.loc[:, ~df.columns.duplicated()]

    df['start_Date'] = pd.to_datetime(df['start_Date'])
    df['start_Year'] = df.start_Date.dt.year
    # df['start_MonthOfYear'] = df.start_Date.dt.month
    # df['start_DayOfWeek'] = df.start_Date.dt.dayofweek
    # df['start_DayOfMonth'] = df.start_Date.dt.day
    df.drop(columns=['start_Date'], inplace=True)
    df = df.reset_index(drop=True)

    return df


station, trip = load_data(PATH, table_names)
trip = generate_features(trip)

print('\n' + 'Preprocessing features and drop unused ones' + '\n')

# train_data_X = trip[['from_station_id', 'start_Year', 'start_MonthOfYear', 'start_DayOfWeek', 'start_DayOfMonth', 'start_HourOfDay']]
train_data_X = trip[['from_station_id', 'start_Year', 'start_DayOfWeek', ]]
train_data_y = trip[['trip_count']]
train_data_X = train_data_X.apply(preprocessing.LabelEncoder().fit_transform)                   # Map Categories to Integer values 0, 1, ...., #categories - 1


print(psm.DataFrameSummary(df).summary())
train_data_X.to_pickle('tmp/train_data_X.pickle')
train_data_y.to_pickle('tmp/train_data_y.pickle')
