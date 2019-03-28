import numpy as np
import pandas as pd
import pandas_summary as pdsm
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.multiclass import unique_labels
from model import *

# X = pd.read_pickle('tmp/train_data_X.pickle').values
# y = pd.read_pickle('tmp/train_data_y.pickle').values
#
# # y_OHE = np.zeros((X.shape[0], len(np.unique(y))))
# # y_OHE[:, 8] = np.ones(X.shape[0])
# # y_OHE = OneHotEncoder(sparse=False).fit_transform(y)
#
# train_ratio = 0.9
# train_size = int(train_ratio * len(X))
#
# X_train = X[:train_size]
# X_val = X[train_size:]
# y_train = y[:train_size]
# y_val = y[train_size:]
# pd.DataFrame(y_val).to_csv('tmp/y_val.csv')
# # y_train = y_OHE[:train_size, :]
# # y_val = y_OHE[train_size:, :]


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


# def evaluate_model(model_loc, X_true, y_true):
#     y_pred = np.array([model_loc.guess(X_true)])
#     y_true = y_true.argmax(1)
#     y_pred = y_pred.reshape((-1, 1))
#     # acc = plot_confusion_matrix(y_true, y_pred)
#     return # acc


def evaluate_model(model_loc, X_true, y_true):
    assert(min(y_true) >= 0)
    y_pred = np.array([model_loc.guess(X_true)])
    relative_err = np.absolute(y_true - y_pred.T)
    result = np.sum(relative_err) / len(y_true)
    return result


def evaluate_model_OHE(model_loc, X_true, y_true):
    y_pred = np.array([model_loc.guess(model_loc.preprocessing_OHE(X_true))])
    relative_err = np.absolute(y_true - y_pred)
    result = np.sum(relative_err) / len(y_true)
    return result


def model_trainer(model_loc, X_tr, y_tr, X_true, y_true, num_iter=1000):
    test_count = count_plots('tmp', 'report.csv')
    model_loc.set_epochs(100)
    for run_iter in range(0, num_iter):
        model_loc.fit(X_tr, y_tr, X_true, y_true)
        res = evaluate_model(model_loc, X_true, y_true)
        os.system(f"echo '{run_iter}. iteration with mae, {str(res)}' >> tmp/{test_count}report.csv")


# model_NNwEE = NNwEE(X_train, y_train, X_val, y_val, epochs_given=200)
# res = evaluate_model(model_NNwEE, X_val, y_val)
# print('rmsle: ' + str(res))
# model_trainer(model_NNwEE, X_train, y_train, X_val, y_val)
