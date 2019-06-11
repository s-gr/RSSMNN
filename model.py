import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from keras.models import Model as KerasModel
from keras.layers import Input, Dense, Activation, Reshape, Embedding, Concatenate, Dropout
from keras.callbacks import ModelCheckpoint
from keras.utils import plot_model
from keras.optimizers import Adam




def plot_history(history):
    loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' not in s]
    val_loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' in s]
    acc_list = [s for s in history.history.keys() if 'acc' in s and 'val' not in s]
    val_acc_list = [s for s in history.history.keys() if 'acc' in s and 'val' in s]

    if len(loss_list) == 0:
        print('Loss is missing in history')
        return

    ## As loss always exists
    epochs = range(1, len(history.history[loss_list[0]]) + 1)

    ## Loss
    plt.figure(1)
    for l in loss_list:
        plt.plot(epochs, history.history[l], 'b', label='Training loss (' + str(str(format(history.history[l][-1], '.5f')) + ')'))
    for l in val_loss_list:
        plt.plot(epochs, history.history[l], 'g', label='Validation loss (' + str(str(format(history.history[l][-1], '.5f')) + ')'))

    plt.grid(True)
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f"plots/{count_plots('plots', 'val_plot.pdf')}th_val_plot.pdf")

    # ## Accuracy
    # plt.figure(2)
    # for l in acc_list:
    #     plt.plot(epochs, history.history[l], 'b', label='Training accuracy (' + str(format(history.history[l][-1], '.5f')) + ')')
    # for l in val_acc_list:
    #     plt.plot(epochs, history.history[l], 'g', label='Validation accuracy (' + str(format(history.history[l][-1], '.5f')) + ')')
    #
    # plt.grid(True)
    # plt.title('Accuracy')
    # plt.xlabel('Epochs')
    # plt.ylabel('Accuracy')
    # plt.legend()
    # plt.savefig(f"plots/{count_plots('plots', 'acc_plot.pdf')}th_acc_plots.pdf")
    plt.show()


def count_plots(myPATH, myEnding):
    num_plots = len([f for f in os.listdir(myPATH) if f.endswith(myEnding) and os.path.isfile(os.path.join(myPATH, f))])
    num_plots += 1
    return num_plots


def split_features(X):
    X_list = []
    for i in range(0, X.shape[1]):
        X_list.append(X[..., [i]])
    return X_list


def split_features_OHE(X):                    # TODO: Dynamically choose number of n_values depending on feature selection.
    # X_OHE = OneHotEncoder(sparse=False, n_values=61).fit_transform(X[..., [0]])
    # X_OHE = np.concatenate((X_OHE, OneHotEncoder(sparse=False, n_values=3).fit_transform(X[..., [1]])), axis=1)
    # X_OHE = np.concatenate((X_OHE, OneHotEncoder(sparse=False, n_values=12).fit_transform(X[..., [2]])), axis=1)
    # X_OHE = np.concatenate((X_OHE, OneHotEncoder(sparse=False, n_values=53).fit_transform(X[..., [3]])), axis=1)
    # X_OHE = np.concatenate((X_OHE, OneHotEncoder(sparse=False, n_values=7).fit_transform(X[..., [4]])), axis=1)
    # X_OHE = np.concatenate((X_OHE, OneHotEncoder(sparse=False, n_values=31).fit_transform(X[..., [5]])), axis=1)
    # X_OHE = np.concatenate((X_OHE, OneHotEncoder(sparse=False, n_values=24).fit_transform(X[..., [6]])), axis=1)
    for clmn in range(0, X.shape[1]):
        if clmn==0:
            X_OHE = OneHotEncoder(sparse=False, n_values=61).fit_transform(X[..., [clmn]])
        else:
            X_OHE = np.concatenate((X_OHE, OneHotEncoder(sparse=False, n_values=len(np.unique(X[:,clmn]))).fit_transform(X[..., [clmn]])), axis=1)
    return X_OHE


class NNwEE:

    def __init__(self, X_train, y_train, X_val, y_val, epochs_given=100):
        self.epochs = epochs_given
        self.dim_inputs = X_train.shape[1]
        self.dim_output = y_train.shape[1]
        self.checkpointer = ModelCheckpoint(filepath="weights/model_weights.hdf5", verbose=1, save_best_only=True)
        self.max_log_y = max(np.max(np.log(y_train)), np.max(np.log(y_val)))
        self.__build_keras_model()
        self.fit(X_train, y_train, X_val, y_val)

    def preprocessing(self, X):
        X = split_features(X)
        return X

    def preprocessing_OHE(self, X):
        X_list = split_features_OHE(X)
        return X_list

    def set_epochs(self, my_epochs):
        self.epochs = my_epochs

    def __build_keras_model(self):
        # input_station = Input(shape=(1,))
        # output_station = Embedding(61, 10, name='station')(input_station)
        # output_station = Reshape(target_shape=(10,))(output_station)
        #
        # input_year = Input(shape=(1,))
        # output_year = Embedding(3, 2, name='year')(input_year)
        # output_year = Reshape(target_shape=(2,))(output_year)
        # #
        # # input_MoY = Input(shape=(1,))
        # # output_MoY = Embedding(12, 4, name='MoY')(input_MoY)
        # # output_MoY = Reshape(target_shape=(4,))(output_MoY)
        # #
        # input_DoW = Input(shape=(1,))
        # output_DoW = Embedding(7, 3, name='DoW')(input_DoW)
        # output_DoW = Reshape(target_shape=(3,))(output_DoW)
        # #
        # # input_DoM = Input(shape=(1,))
        # # output_DoM = Embedding(31, 10, name='DoM')(input_DoM)
        # # output_DoM = Reshape(target_shape=(10,))(output_DoM)
        # #
        # # input_HoD = Input(shape=(1,))
        # # output_HoD = Embedding(24, 10, name='HoD')(input_HoD)
        # # output_HoD = Reshape(target_shape=(10,))(output_HoD)
        # #
        # input_model = [input_station,
        #                # input_MoY,
        #                input_year,
        #                input_DoW  # ,
        #                # input_DoM,
        #                # input_HoD
        #                ]
        # output_model = [output_station,
        #                 # output_MoY,
        #                 output_year,
        #                 output_DoW  # ,
        #                 # output_DoM,
        #                 # output_HoD
        #                 ]
        #
        # # output_model = Concatenate()(output_model)
        input_model = Input(shape=(191, ))
        output_model = Dense(1000, kernel_initializer="random_uniform")(input_model)
        output_model = Activation('sigmoid')(output_model)
        output_model = Dropout(0.5)(output_model)
        output_model = Dense(1000, kernel_initializer="random_uniform")(output_model)
        output_model = Activation('sigmoid')(output_model)
        output_model = Dropout(0.5)(output_model)
        output_model = Dense(1000, kernel_initializer="random_uniform")(output_model)
        output_model = Activation('sigmoid')(output_model)
        output_model = Dropout(0.5)(output_model)
        output_model = Dense(self.dim_output)(output_model)
        # output_model = Activation('softmax')(output_model)

        self.model = KerasModel(inputs=input_model, outputs=output_model)
        self.model.compile(loss='mean_absolute_error',  # 'categorical_crossentropy',
                           # optimizer='adam' #,
                           optimizer=Adam(lr=0.001)  # ,
                           # metrics=['accuracy']
                           )
        if os.path.isfile('/weights/model_weights.hdf5'):
            self.model.load_weights('weights/model_weights.hdf5')
        plot_model(self.model, show_shapes=True, show_layer_names=True, rankdir='LR', to_file='tmp/model.png')
        self.model.summary()

    def _val_for_fit(self, val):  # Set y-value to log(y) for fitting
        val = np.log(val) / self.max_log_y
        return val

    def _val_for_pred(self, val):  # back transform to non log value of y
        return np.exp(val * self.max_log_y)

    def fit(self, X_train, y_train, X_val, y_val):
        history = self.model.fit(self.preprocessing_OHE(X_train), y_train,
                                 validation_data=(self.preprocessing_OHE(X_val), y_val),
                                 epochs=self.epochs, batch_size=128, callbacks=[self.checkpointer]
                                 )
        # history = self.model.fit(X_train, y_train,
        #                          validation_data=(X_val, y_val),
        #                          epochs=self.epochs, batch_size=128, callbacks=[self.checkpointer]
        #                          )
        self.model.save_weights('weights/model_weights.hdf5')
        plot_history(history)

    def guess(self, features):
        features = self.preprocessing_OHE(features)
        y_pred = self.model.predict(features).flatten()
        pd.DataFrame(np.rint(y_pred).astype(int)).to_csv('tmp/y_pred.csv')
        return np.rint(y_pred)


    # def guess(self, features):
    #     features = self.preprocessing(features)
    #     result = self.model.predict(features)
    #     result = result.argmax(1)
    #     pd.DataFrame(result).to_csv('tmp/y_pred.csv')
    #     return result

# un