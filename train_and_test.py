import pickle
import numpy
from model import *
import pandas as pd

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
    indices = numpy.random.randint(num_row, size=n)
    return X[indices, :], y[indices]


X_train, y_train = sample(X_train, y_train, 400000)  # Simulate data sparsity
print("Number of samples used for training: " + str(y_train.shape[0]))

print("Fitting NNwEE...")
model = NNwEE(X_train, y_train, X_val, y_val)


def evaluate_model(model_loc, X, y):
    assert(min(y) > 0)
    guessed_sales = numpy.array([model_loc.guess(X)])
    mean_sales = guessed_sales.mean(axis=0)
    relative_err = numpy.absolute((y - mean_sales) / y)
    result = numpy.sum(relative_err) / len(y)
    return result


print("Evaluate combined models...")
print("Training error...")
r_train = evaluate_model(model, X_train, y_train)
print(r_train)

print("Validation error...")
r_val = evaluate_model(model, X_val, y_val)
print(r_val)
