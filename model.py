import numpy as np
import matplotlib.pyplot as plt
from keras.models import Model as KerasModel
from keras.layers import Input, Dense, Activation, Reshape, Embedding, Concatenate
from keras.callbacks import ModelCheckpoint
from keras.utils import plot_model


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


def split_features(X):                                                                      # Take cats and split them into a "vector"
    X_list = []                                                                             # 0-th column is just 0s since we only allowd 'Open' stores
    for i in range(1, 9):
        X_list.append(X[..., [i]])
    return X_list


class NNwEE:

    def __init__(self, X_train, y_train, X_val, y_val):
        # super().__init__()
        self.epochs = 10
        self.checkpointer = ModelCheckpoint(filepath="best_model_weights.hdf5", verbose=1, save_best_only=True)
        self.max_log_y = max(np.max(np.log(y_train)), np.max(np.log(y_val)))
        self.__build_keras_model()
        self.fit(X_train, y_train, X_val, y_val)

    def preprocessing(self, X):
        X_list = split_features(X)
        return X_list

    def __build_keras_model(self):                                                          # Build Model. Starting with individual Embeddings for every cat. Using Keras
                                                                                            # Functional API
        input_store = Input(shape=(1,))
        output_store = Embedding(1115, 10, name='store_embedding')(input_store)
        output_store = Reshape(target_shape=(10,))(output_store)

        input_dow = Input(shape=(1,))
        output_dow = Embedding(7, 6, name='dow_embedding')(input_dow)
        output_dow = Reshape(target_shape=(6,))(output_dow)

        input_promo = Input(shape=(1,))
        output_promo = Dense(1)(input_promo)

        input_year = Input(shape=(1,))
        output_year = Embedding(3, 2, name='year_embedding')(input_year)
        output_year = Reshape(target_shape=(2,))(output_year)

        input_month = Input(shape=(1,))
        output_month = Embedding(12, 6, name='month_embedding')(input_month)
        output_month = Reshape(target_shape=(6,))(output_month)

        input_day = Input(shape=(1,))
        output_day = Embedding(31, 10, name='day_embedding')(input_day)
        output_day = Reshape(target_shape=(10,))(output_day)

        input_germanstate = Input(shape=(1,))
        output_germanstate = Embedding(12, 6, name='state_embedding')(input_germanstate)
        output_germanstate = Reshape(target_shape=(6,))(output_germanstate)

        input_holiday = Input(shape=(1,))
        output_holiday = Dense(1)(input_holiday)

        input_model = [input_store, input_dow, input_promo,
                       input_year, input_month, input_day, input_germanstate, input_holiday]

        output_embeddings = [output_store, output_dow, output_promo,
                             output_year, output_month, output_day, output_germanstate, output_holiday]

        output_model = Concatenate()(output_embeddings)                                     # Concatenate inputs to model
        output_model = Dense(1000, kernel_initializer="uniform")(output_model)
        output_model = Activation('relu')(output_model)
        output_model = Dense(500, kernel_initializer="uniform")(output_model)
        output_model = Activation('relu')(output_model)
        output_model = Dense(1)(output_model)
        output_model = Activation('sigmoid')(output_model)

        self.model = KerasModel(inputs=input_model, outputs=output_model)

        self.model.compile(loss='mean_absolute_error', optimizer='adam')
        self.model.summary()
        plot_model(self.model, show_shapes=True, show_layer_names=True, rankdir = 'LR', to_file='tmp/model.png')
        self.model.compile(loss='mean_absolute_error', optimizer='adam')

    def evaluate(self, X_val, y_val):
        assert(min(y_val) > 0)
        guessed_sales = self.guess(X_val)
        relative_err = np.absolute((y_val - guessed_sales) / y_val)
        result = np.sum(relative_err) / len(y_val)
        return result

    def _val_for_fit(self, val):                                                                                # Set y-value to log(y) for fitting
        val = np.log(val) / self.max_log_y
        return val

    def _val_for_pred(self, val):                                                                               # back transform to non log value of y
        return np.exp(val * self.max_log_y)

    def fit(self, X_train, y_train, X_val, y_val):
        history = self.model.fit(self.preprocessing(X_train), self._val_for_fit(y_train),
                       validation_data=(self.preprocessing(X_val), self._val_for_fit(y_val)),
                       epochs=self.epochs, batch_size=128,
                       # callbacks=[self.checkpointer],
                       )
        # self.model.load_weights('best_model_weights.hdf5')
        plot_history(history)
        print("Result on validation data: ", self.evaluate(X_val, y_val))

    def guess(self, features):
        features = self.preprocessing(features)
        result = self.model.predict(features).flatten()
        return self._val_for_pred(result)
