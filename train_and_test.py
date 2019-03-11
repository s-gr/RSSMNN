import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.multiclass import unique_labels
from model import *

X = pd.read_pickle('tmp/train_data_X.pickle').values
y = pd.read_pickle('tmp/train_data_y.pickle').values
pd.DataFrame(y).to_csv('tmp/y_val.csv')


y_OHE = np.zeros((X.shape[0], 20))
y_OHE[:, 8] = np.ones(X.shape[0])
y_OHE = preprocessing.OneHotEncoder(sparse=False).fit_transform(y)

train_ratio = 0.9
train_size = int(train_ratio * len(X))

X_train = X[:train_size]
X_val = X[train_size:]
# y_train = y[:train_size]
# y_val = y[train_size:]
y_train = y_OHE[:train_size, :]
y_val = y_OHE[train_size:, :]

def plot_confusion_matrix(y_true, y_val,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_val)
    acc = cm.trace() / cm.sum() if cm.sum != 0 else -1
    # prec = cm.trace() / cm.sum() if cm.sum != 0 else -1
    # rec = cm.trace() / cm.sum() if cm.sum != 0 else -1
    # Only use the labels that appear in the data
    classes = np.char.mod('%d', unique_labels(y_true, y_val))
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

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
    plt.setp(ax.get_xticklabels(), rotation=0, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt), fontsize=2,
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    plt.savefig(f"plots/{count_plots('plots', 'conf_mat.pdf')}th_conf_mat.pdf")
    plt.show()
    return acc  # , prec, rec


def evaluate_model(model_loc, X_true, y_true):
    y_pred = np.array([model_loc.guess(X_true)])
    y_true = y_true.argmax(1)
    y_pred = y_pred.reshape((-1, 1))
    acc = plot_confusion_matrix(y_true, y_pred)
    return acc


def model_trainer(model_loc, X_tr, y_tr, X_true, y_true, num_iter=1000):
    test_count = count_plots('tmp', 'report.csv')
    model_loc.set_epochs(100)
    for run_iter in range(0, num_iter):
        model_loc.fit(X_tr, y_tr, X_true, y_true)
        acc = evaluate_model(model_NNwEE, X_true, y_true)
        os.system(f"echo '{run_iter}. iteration with acc, {str(acc)}' >> tmp/{test_count}report.csv")


model_NNwEE = NNwEE(X_train, y_train, X_val, y_val, epochs_given=1)
model_trainer(model_NNwEE, X_train, y_train, X_val, y_val)
