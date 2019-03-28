import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_summary as pdsm

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from model import *

X = pd.read_pickle('tmp/train_data_X.pickle').values
y = pd.read_pickle('tmp/train_data_y.pickle').values

df_X = pd.read_pickle('tmp/train_data_X.pickle')
df_y = pd.read_pickle('tmp/train_data_y.pickle')

