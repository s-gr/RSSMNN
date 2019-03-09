import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras.models import Model as KerasModel
from keras.layers import Input, Dense, Activation, Reshape, Embedding, Concatenate, Dropout, BatchNormalization
from keras.callbacks import ModelCheckpoint
from keras.utils import plot_model
from keras import regularizers
import keras as krs


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
    plt.savefig(f"plots/{count_plots('plots')}_th_of_my_val_plots.pdf")

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
    plt.show()


def count_plots(myPATH):
    num_plots = len([f for f in os.listdir(myPATH) if f.endswith('.pdf') and os.path.isfile(os.path.join(myPATH, f))])
    num_plots += 1
    return num_plots


def split_features(X):                                                                      # Take cats and split them into a "vector"
    X_list = []                                                                             # 0-th column is just 0s since we only allowed 'Open' stores
    for i in range(0, 6):
        X_list.append(X[..., [i]])
    return X_list


class NNwEE:

    def __init__(self, X_train, y_train, X_val, y_val):
        # super().__init__()
        self.epochs = 20
        # self.checkpointer = ModelCheckpoint(filepath="best_model_weights.hdf5", verbose=1, save_best_only=True)
        self.max_log_y = max(np.max(np.log(y_train)), np.max(np.log(y_val)))
        self.__build_keras_model()
        self.fit(X_train, y_train, X_val, y_val)

    def preprocessing(self, X):
        X_list = split_features(X)
        return X_list

    def __build_keras_model(self):                                                          # Build Model. Starting with individual Embeddings for every cat. Using Keras
                                                                                            # Functional API
        input_station = Input(shape=(1,))
        output_station = Embedding(60, 10, name='station_embedding')(input_station)
        output_station = Reshape(target_shape=(10,))(output_station)

        input_year = Input(shape=(1,))
        output_year = Embedding(3, 2, name='year_embedding')(input_year)
        output_year = Reshape(target_shape=(2,))(output_year)

        input_MoY = Input(shape=(1,))
        output_MoY = Embedding(12, 4, name='MoY_embedding')(input_MoY)
        output_MoY = Reshape(target_shape=(4,))(output_MoY)

        input_DoW = Input(shape=(1,))
        output_DoW = Embedding(7, 3, name='DoW_embedding')(input_DoW)
        output_DoW = Reshape(target_shape=(3,))(output_DoW)

        input_DoM = Input(shape=(1,))
        output_DoM = Embedding(31, 10, name='DoM_embedding')(input_DoM)
        output_DoM = Reshape(target_shape=(10,))(output_DoM)

        input_HoD = Input(shape=(1,))
        output_HoD = Embedding(24, 10, name='HoD_embedding')(input_HoD)
        output_HoD = Reshape(target_shape=(10,))(output_HoD)

        input_model = [input_station, input_MoY, input_year, input_DoW, input_DoM, input_HoD]
        output_embeddings = [output_station, output_MoY, output_year, output_DoW, output_DoM, output_HoD]

        output_model = Concatenate()(output_embeddings)                                                         # Concatenate inputs to model
        output_model = Dense(1000, kernel_initializer="uniform")(output_model)
        # output_model = BatchNormalization()(output_model)
        output_model = Activation('relu')(output_model)
        output_model = Dropout(0.5)(output_model)
        output_model = Dense(500, kernel_initializer="uniform")(output_model)
        # output_model = BatchNormalization()(output_model)
        output_model = Activation('relu')(output_model)
        output_model = Dropout(0.5)(output_model)
        output_model = Dense(1)(output_model)
        output_model = BatchNormalization()(output_model)
        output_model = Activation('sigmoid')(output_model)

        self.model = KerasModel(inputs=input_model, outputs=output_model)

        self.model.compile(loss='mean_absolute_error', optimizer=krs.optimizers.Adam(lr=0.001))    # , optimizer='adam')
        self.model.summary()
        plot_model(self.model, show_shapes=True, show_layer_names=True, rankdir = 'LR', to_file='tmp/model.png')

    def evaluate(self, X_val, y_val):
        assert(min(y_val) > 0)
        guessed_demand = self.guess(X_val)
        guessed_demand = guessed_demand.reshape(guessed_demand.size, 1)
        pd.DataFrame(guessed_demand).to_csv('tmp/gd.csv')
        pd.DataFrame(y_val).to_csv('tmp/v_val.csv')
        relative_err = np.absolute((y_val - guessed_demand))  # / y_val)
        result = np.sum(relative_err) / y_val.size
        return result

    def _val_for_fit(self, val):                                                                                # Set y-value to log(y) for fitting
        val = np.log(val) / self.max_log_y
        return val

    def _val_for_pred(self, val):                                                                               # back transform to non log value of y
        return np.exp(val * self.max_log_y)

    def fit(self, X_train, y_train, X_val, y_val):
        history = self.model.fit(self.preprocessing(X_train), self._val_for_fit(y_train),
                                 validation_data=(self.preprocessing(X_val), self._val_for_fit(y_val)),
                                 epochs=self.epochs, batch_size=128) #,callbacks=[self.checkpointer]
        # self.model.load_weights('best_model_weights.hdf5')
        plot_history(history)
        print("Result on validation data: ", self.evaluate(X_val, y_val))
    # def fit(self, X_train, y_train, X_val, y_val):
    #     history = self.model.fit(self.preprocessing(X_train), y_train,
    #                              validation_data=(self.preprocessing(X_val), y_val),
    #                              epochs=self.epochs, batch_size=128) #,callbacks=[self.checkpointer]
    #     # self.model.load_weights('best_model_weights.hdf5')
    #     plot_history(history)
    #     print("Result on validation data: ", self.evaluate(X_val, y_val))

    def guess(self, features):
        features = self.preprocessing(features)
        result = self.model.predict(features).flatten()
        pd.DataFrame(result).to_csv('tmp/gd_train.csv')
        return self._val_for_pred(result)
