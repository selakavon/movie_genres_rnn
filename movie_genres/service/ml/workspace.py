"""Training and Scoring Workspace module."""

from pathlib import Path
import pandas as pd

from movie_genres.service.ml.configuration import TrainConfiguration, \
    EmbeddingConfiguration
from movie_genres.service.ml.model import Model
from movie_genres.service.ml.text_processing import TextProcessor, Embeddings
from movie_genres.service.ml.persistence_util import dump_object, load_object
import logging


class WorkSpace():
    """
    Workspace class.

    Workspace class encapsulates text processing, training and scoring
    and provides api for saving and loading the state.
    """

    __TEXT_PROCESSOR_FILE = "text_processor.pickle"

    def __init__(self, name: str):
        self.__name = name
        self.__logger = logging.getLogger(self.id)

    @property
    def id(self):
        """Return workspace identifier."""
        return f"{type(self).__name__}({self.__name})"

    def train(self, df: pd.DataFrame,
              train_config: TrainConfiguration,
              embedding_config: EmbeddingConfiguration) -> None:
        """
        Process input and labels, performs training.

        :param df: input
        :param train_config: training configuration
        :param embedding_config: embedding configuration
        """
        self.__logger.info("train")

        x_train = self.__getX(df)
        y_train = self.__getY(df)

        self.__text_processor = TextProcessor(
            embedding_config.max_len,
            embedding_config.max_words
        )

        x_processed = self.__text_processor(x_train)

        self.__embeddings = Embeddings(embedding_config).read_embeddings(
            self.__text_processor.word_index
        )

        model = Model()

        model.fit(
            x_processed, y_train,
            embedding_config, train_config, self.__embeddings.embedding_matrix
        )

        self.__model = model

    def predict(self, df):
        """
        Process input and labels, performs scoring.

        :param df: input
        :return: scores dataframe
        """
        x_test = self.__getX(df)

        x_test_seq_pad = self.__text_processor(x_test, test=True)

        predictions = self.__model.predict(x_test_seq_pad)

        return pd.DataFrame(
            {
                "movie_id": df["movie_id"],
                "predicted_genres": predictions
            }
        )

    def save(self):
        """Save workspace."""
        self.__model_path.mkdir(parents=True, exist_ok=True)

        dump_object(self.__text_processor,
                    self.__workspace_path / self.__TEXT_PROCESSOR_FILE)

        self.__model.save(self.__model_path)

    def load(self):
        """Load workspace."""
        self.__text_processor = load_object(
            self.__workspace_path / self.__TEXT_PROCESSOR_FILE
        )

        model = Model()
        model.load(self.__model_path)
        self.__model = model

    @property
    def __workspace_path(self):
        return Path(f"workspace/{self.__name}/model")

    @property
    def __model_path(self):
        return self.__workspace_path / "model"

    def __getX(self, df):
        return df["synopsis"].str.lower()

    def __getY(self, df):
        return df["genres"].str.get_dummies(sep=" ")
