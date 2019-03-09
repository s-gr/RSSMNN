import numpy as np
import pandas as pd
from model import *

X = pd.read_pickle('tmp/train_data_X.pickle').values
y = pd.read_pickle('tmp/train_data_y.pickle').values
train_ratio = 0.9
train_size = int(train_ratio * len(X))

X_train = X[:train_size]
X_val = X[train_size:]
y_train = y[:train_size]
y_val = y[train_size:]

# print(y_train.size)
# print(y_val.size)

# plt.figure()
# plt.hist(y_train, bins=44)
# plt.show()
#
# plt.figure()
# plt.hist(y_val, bins=44)
# plt.show()

def sample(X, y, n):
    '''random samples'''
    num_row = X.shape[0]
    indices = np.random.randint(num_row, size=n)
    return X[indices, :], y[indices]


X_train, y_train = sample(X_train, y_train, 100000)  # Simulate data sparsity
print("Number of samples used for training: " + str(y_train.shape[0]))

print("Fitting NNwEE...")
model = NNwEE(X_train, y_train, X_val, y_val)


def evaluate_model(model_loc, X, y):
    assert(min(y) > 0)
    guessed_demand = np.array([model_loc.guess(X)])
    print(guessed_demand.shape)
    guessed_demand = guessed_demand.reshape(guessed_demand.size, 1)
    relative_err = np.absolute((y - guessed_demand)) # / y)
    result = np.sum(relative_err) / len(y)
    return result


r_train = evaluate_model(model, X_train, y_train)
print('Result on training data: ' + str(r_train))
