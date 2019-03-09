import numpy as np
import pandas as pd
from model import *

train_ratio = 0.9

(X, y) = (pd.read_pickle('tmp/train_data_X.pickle').values, pd.read_pickle('tmp/train_data_y.pickle').values)

train_size = int(train_ratio * len(X))

X_train = X[:train_size]
X_val = X[train_size:]
y_train = y[:train_size]
y_val = y[train_size:]


def sample(X, y, n):
    '''random samples'''
    num_row = X.shape[0]
    indices = np.random.randint(num_row, size=n)
    return X[indices, :], y[indices]


X_train, y_train = sample(X_train, y_train, 200)  # Simulate data sparsity
print("Number of samples used for training: " + str(y_train.shape[0]))

print("Fitting NNwEE...")
model = NNwEE(X_train, y_train, X_val, y_val)


def evaluate_model(model_loc, X, y):
    assert(min(y) > 0)
    guessed_demand = np.array([model_loc.guess(X)])
    print('safe variables')
    X.tofile('X_saved')
    y.tofile('y_saved')
    guessed_demand.tofile('gd_saved')
    mean_demand = guessed_demand.mean(axis=0)
    relative_err = np.absolute((y - mean_demand) / y)
    result = np.sum(relative_err) / len(y)
    return result


print("Evaluate combined models...")
print("Training error...")
r_train = evaluate_model(model, X_train, y_train)
print(r_train)

print("Validation error...")
r_val = evaluate_model(model, X_val, y_val)
print(r_val)
