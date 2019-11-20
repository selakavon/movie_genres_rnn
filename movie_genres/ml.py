"""Main entry poin to training and predicting backend."""

from movie_genres.service.ml.configuration import TrainConfiguration, \
    EmbeddingConfiguration
from movie_genres.service.ml.workspace import WorkSpace


def train(df):
    """Create new workspace, train model and save it."""
    workspace = WorkSpace("1")

    workspace.train(df, TrainConfiguration(), EmbeddingConfiguration())

    workspace.save()


def predict(df):
    """Load workspace and return predictions."""
    workspace = WorkSpace("1")
    workspace.load()

    pred_df = workspace.predict(df)

    return pred_df
