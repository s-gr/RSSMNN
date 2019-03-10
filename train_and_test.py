import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.multiclass import unique_labels
from model import *

X = pd.read_pickle('tmp/train_data_X.pickle').values
y = pd.read_pickle('tmp/train_data_y.pickle').values       # shape (len(y), 1)
pd.DataFrame(y).to_csv('tmp/y_val.csv')
# y_OHE = np.zeros((X.shape[0], 20))
# y_OHE[:, 8] = np.ones(X.shape[0])

y_OHE = preprocessing.OneHotEncoder(sparse=False).fit_transform(y)   # shape (len(y), 20)

train_ratio = 0.9
train_size = int(train_ratio * len(X))

X_train = X[:train_size]
X_val = X[train_size:]
y_train = y_OHE[:train_size, :]
y_val = y_OHE[train_size:, :]


def plot_confusion_matrix(y_true, y_val,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_val)
    # Only use the labels that appear in the data
    classes = np.char.mod('%d', unique_labels(y_true, y_val))
    # classes =
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.figure()
    ax = plt.gca()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    plt.show()


def evaluate_model(model_loc, X_true, y_true):
    y_pred = np.array([model_loc.guess(X_true)])
    y_true = y_true.argmax(1)
    y_pred = y_pred.reshape((-1, 1))
    plot_confusion_matrix(y_true, y_pred)
    unique, counts = np.unique(y_pred, return_counts=True)
    return unique, counts


model = NNwEE(X_train, y_train, X_val, y_val, epochs_given=10)

print(evaluate_model(model, X_val, y_val))
