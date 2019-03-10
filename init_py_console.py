import numpy as np
import pandas as pd
from model import *

X = pd.read_pickle('tmp/train_data_X.pickle').values
y = pd.read_pickle('tmp/train_data_y.pickle').values