import numpy as np
from sklearn.utils import shuffle
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    CSVLogger,
)
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.models import Sequential
from tqdm.notebook import tnrange

from util import Const


class Model:
    def __init__(self):
        self._model = self._build_model()
        self._callbacks = [
            ModelCheckpoint(
                filepath=Const.BEST_WEIGHTS_FILE_PATH,
                save_best_only=True,
                save_weights_only=True,
                monitor='val_rmse',
                mode='min',
            ),
            CSVLogger(
                filename=Const.HISTORY_FILE_PATH,
                append=True,
            )
        ]
        self._X_train, self._Y_train, self._X_val, self._Y_val = None, None, None, None

    @staticmethod
    def _build_model():
        model = Sequential()

        model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same',
                         input_shape=(Const.YANDEX_MAPS_MAX_HEIGHT, Const.YANDEX_MAPS_MAX_WIDTH, 3)))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dense(1))

        model.compile(
            optimizer='sgd',
            loss='mse',
            metrics=[RootMeanSquaredError(name='rmse')]
        )
        model.summary()

        return model

    def set_data(self, X: list, Y: [float]):
        self._X_train, self._Y_train, self._X_val, self._Y_val = self._split_data(*self._preprocess_data(X, Y))

    @staticmethod
    def preprocess(data):
        return preprocess_input(data)

    @staticmethod
    def _preprocess_data(X, Y):
        tmp_X = np.array([preprocess_input(X[i]) for i in tnrange(len(X))])
        tmp_Y = np.array(Y).astype('float32')

        return tmp_X, tmp_Y

    @staticmethod
    def _split_data(X, Y):
        N = min(len(X), len(Y))
        validation_size = int(N * Const.VALIDATION_SPLIT)

        X_train, Y_train = shuffle(X, Y, random_state=Const.RANDOM_SEED)
        X_val, Y_val = X_train[:validation_size], Y_train[:validation_size]
        X_train, Y_train = X_train[validation_size:N], Y_train[validation_size:N]

        return X_train, Y_train, X_val, Y_val

    def train(self, epochs):
        if self._X_train is None or self._Y_train is None or self._X_val is None or self._Y_val is None:
            raise RuntimeError('You should set the data before starting to train. Use `model.data = (X, Y)` first')

        history = self._model.fit(
            self._X_train,
            self._Y_train,
            validation_data=(
                self._X_val,
                self._Y_val,
            ),
            batch_size=8,
            epochs=epochs,
            callbacks=self._callbacks,
        )

        return history
