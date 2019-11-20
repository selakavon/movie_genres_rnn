"""Module for text processing and word embeedding."""

import numpy as np
import tensorflow as tf

from movie_genres.service.ml.configuration import EmbeddingConfiguration
import logging


class TextProcessor():
    """Text processing and padding."""

    def __init__(self, max_len, max_words):
        self.__max_len = max_len
        self.__max_words = max_words
        self.__logger = logging.getLogger(type(self).__name__)

    def __call__(self, X, test=False):
        """
        Process input texts.

        :param X: input
        :param test: flag for test or train dataset
        :return: processed texts
        """
        self.__logger.debug("TextProcessor")
        if not test:
            self.__fit_tokenizer(X)

        x_train_seq = self.__tokenize(X)
        x_train_seq_pad = self.__pad(x_train_seq)

        return x_train_seq_pad

    def __tokenize(self, X):
        self.__logger.debug("__tokenize")
        return self.__tokenizer.texts_to_sequences(X)

    def __pad(self, X):
        self.__logger.debug("__pad")
        x_train_seq_pad = tf.keras.preprocessing.sequence.pad_sequences(
            X, maxlen=self.__max_len)

        return x_train_seq_pad

    def __fit_tokenizer(self, X):
        self.__logger.debug("__fit_tokenizer")
        tokenizer = tf.keras.preprocessing.text.Tokenizer(
            num_words=self.__max_words, lower=True)
        tokenizer.fit_on_texts(X)

        self.__tokenizer = tokenizer

    @property
    def word_index(self):
        """Tokenizer's word index created by fitting on train dataset."""
        return self.__tokenizer.word_index


class Embeddings():
    """Word Embedding."""

    def __init__(self, config: EmbeddingConfiguration):
        self.__config = config
        self.__logger = logging.getLogger(type(self).__name__)

    def read_embeddings(self, word_index):
        """
        Read and parse embeddings.

        :param word_index: Tokenizer's word index.
        :return: self instance
        """
        self.__logger.debug("read_embeddings")

        embeddings_index = {}

        with open(self.__config.embed_file, encoding='utf8') as f:
            for line in f:
                values = line.rstrip().rsplit(' ')
                word = values[0]
                embed = np.asarray(values[1:], dtype='float32')
                embeddings_index[word] = embed

        num_words = min(self.__config.max_words, len(word_index) + 1)

        embedding_matrix = np.zeros(
            (num_words, self.__config.embed_size),
            dtype='float32'
        )

        for word, i in word_index.items():

            if i >= self.__config.max_words:
                continue

            embedding_vector = embeddings_index.get(word)

            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector

        self.__embedding_matrix = embedding_matrix

        return self

    @property
    def embedding_matrix(self):
        """Return embedding matrix."""
        return self.__embedding_matrix
