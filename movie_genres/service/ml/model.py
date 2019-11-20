"""Model wrapper module."""

import tensorflow as tf
import numpy as np
import pandas as pd
from .persistence_util import dump_object, load_object

from movie_genres.ml import EmbeddingConfiguration, TrainConfiguration

from pathlib import Path
import logging


class Model():
    """Keras model implementation class."""

    __MODEL_WEIGHTS_FILE = "model_weights"
    __EMBEDDING_CONFIG = "embedding_config.pickle"
    __TRAIN_CONFIG = "traing_config.pickle"
    __EMBEDDING_MATRIX = "embedding_matrix.pickle"
    __GENRE_NAMES = "genre_names.pickle"

    def __init__(self):
        self.__logger = logging.getLogger(type(self).__name__)

    def __get_model(self, label_count, max_len, embedding_matrix):
        """
        Create model architecture.

        :param label_count: Cound of unqiue genres.
        :param max_len: Maximum length of all sequences
        :param embedding_matrix: Embedding matrix.
        :return: model
        """
        self.__logger.debug("get_model")

        input = tf.keras.layers.Input(shape=(max_len,))

        x = tf.keras.layers.Embedding(
            embedding_matrix.shape[0],
            embedding_matrix.shape[1],
            weights=[embedding_matrix],
            trainable=False)(input)

        x = tf.keras.layers.Bidirectional(
            tf.keras.layers.GRU(
                128, return_sequences=True, dropout=0.1, recurrent_dropout=0.1
            ))(x)

        x = tf.keras.layers.Conv1D(
            64, kernel_size=3, padding="valid",
            kernel_initializer="glorot_uniform")(x)

        avg_pool = tf.keras.layers.GlobalAveragePooling1D()(x)
        max_pool = tf.keras.layers.GlobalMaxPooling1D()(x)

        x = tf.keras.layers.concatenate([avg_pool, max_pool])

        preds = tf.keras.layers.Dense(label_count, activation="sigmoid")(x)

        model = tf.keras.Model(input, preds)

        model.summary(print_fn=self.__logger.debug)

        model.compile(
            loss='binary_crossentropy',
            optimizer=tf.keras.optimizers.Adam(lr=1e-3), metrics=['accuracy']
        )

        return model

    def fit(self, X: np.ndarray, y: pd.DataFrame,
            embedding_config: EmbeddingConfiguration,
            train_config: TrainConfiguration,
            embedding_matrix: np.array) -> "Model":
        """
        Train model.

        :param X: input
        :param y: labels
        :param embedding_config: Embedding configuration.
        :param train_config: Training configuration.
        :param embedding_matrix: Embedding matrix.
        :return:
        """
        model = self.__get_model(
            y.shape[1], embedding_config.max_len, embedding_matrix
        )

        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                patience=train_config.early_stop, monitor='val_loss'
            )
        ]

        model.fit(X, y.values, validation_split=train_config.validation_split,
                  batch_size=train_config.batch_size,
                  epochs=train_config.epochs, callbacks=callbacks, verbose=1)

        self.__model = model
        self.__genre_names = y.columns

        self.__embedding_config = embedding_config
        self.__train_config = train_config
        self.__embedding_matrix = embedding_matrix

        return self

    def predict(self, X):
        """
        Scoring.

        :param X: input
        :return: scores dataframe
        """
        self.__logger.debug("predict")

        predictions = self.__model.predict(X)

        def top_five(pred):
            return np.str.join(" ", self.__genre_names[(-pred).argsort()[:5]])
        pred_np = np.apply_along_axis(top_five, 1, predictions)

        return pred_np

    def save(self, path: Path) -> None:
        """
        Load model and configuration.

        :param path: base directory
        """
        path.mkdir(parents=True, exist_ok=True)

        self.__save_training_meta(path)
        self.__model.save_weights(str(path / self.__MODEL_WEIGHTS_FILE))

    def load(self, path: Path) -> None:
        """
        Load model and configuration.

        :param path: base directory
        """
        self.__load_training_meta(path)

        model = self.__get_model(
            len(self.__genre_names),
            self.__embedding_config.max_len,
            self.__embedding_matrix
        )
        model.load_weights(str(path / self.__MODEL_WEIGHTS_FILE))

        self.__model = model

    def __save_training_meta(self, path):

        dump_object(self.__embedding_config,
                    path / self.__EMBEDDING_CONFIG)
        dump_object(self.__train_config,
                    path / self.__TRAIN_CONFIG)
        dump_object(self.__embedding_matrix,
                    path / self.__EMBEDDING_MATRIX)
        dump_object(self.__genre_names,
                    path / self.__GENRE_NAMES)

    def __load_training_meta(self, path):

        self.__embedding_config = load_object(path / self.__EMBEDDING_CONFIG)
        self.__train_config = load_object(path / self.__TRAIN_CONFIG)
        self.__embedding_matrix = load_object(path / self.__EMBEDDING_MATRIX)
        self.__genre_names = load_object(path / self.__GENRE_NAMES)
