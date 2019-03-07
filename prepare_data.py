import pandas as pd
import pandas_summary as psm
from sklearn import preprocessing

PATH = 'dataset/'  # set path variable for dataset
table_names = ['station', 'new_trip']


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
        if 'StateHoliday' in table.columns:                                                     # Simplify State Holiday
            table.StateHoliday = (table.StateHoliday != '0')
        # if 'SchoolHoliday' in table.columns:                                                    # Simplify State Holiday
        #     table['SchoolHoliday'] = (table['SchoolHoliday'] != '0')
        print(table.head())
    return tables


def generate_features(joined_df):                                                               # Generate new features
    joined_df['Year'] = joined_df.Date.dt.year
    joined_df['Month'] = joined_df.Date.dt.month
    joined_df['Day'] = joined_df.Date.dt.day


def join_dfs(left, right, left_on, right_on = None, suffix = '_y'):                             # left-join two data frames
    if right_on is None:
        right_on = left_on
    return left.merge(right, how='left', left_on=left_on, right_on=right_on, suffixes=("", suffix))


train, store, store_states, test = load_data(PATH, table_names)                                 # Load tables to data frames
train.sort_values(['Date', 'Store'], ascending = True, inplace=True)

joined = join_dfs(train, store, 'Store')                                                        # Join train and store set at Store column
joined = join_dfs(joined, store_states, 'Store')                                                # Join joined and store_states at Store column

generate_features(joined)

print(joined.head().T)
print(psm.DataFrameSummary(joined).summary().T)
print('\n' + 'Preprocessing features and drop unused ones' + '\n')

joined = joined.loc[(joined['Sales'] != 0) & (joined['Open'] != 0)]
train_data_X = joined[['Open', 'Store', 'DayOfWeek', 'Promo', 'Year', 'Month', 'Day', 'State', 'StateHoliday']]
train_data_y = joined['Sales']
train_data_X = train_data_X.apply(preprocessing.LabelEncoder().fit_transform)                   # Map Categories to Integer values 0, 1, ...., #categories - 1

print(psm.DataFrameSummary(train_data_X).summary())
print(train_data_X.head())

train_data_X.to_pickle('tmp/train_data_X.pickle')
train_data_y.to_pickle('tmp/train_data_y.pickle')
