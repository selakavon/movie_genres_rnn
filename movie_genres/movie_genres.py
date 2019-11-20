"""Application and routes module."""

import connexion
from pkg_resources import resource_filename

from movie_genres.api.predict.predict_controller import PredictController
from movie_genres.api.train.train_controller import TrainController
import logging
import logging.config
import yaml

train_controller = TrainController()
predict_controller = PredictController()


def train_post(body):
    """Train POST route."""
    return train_controller.post(body)


def predict_post(body):
    """Predict POST route."""
    return predict_controller.post(body)


def create_app():
    """Create flask application."""
    logging_conf_file = resource_filename("movie_genres", "logging.conf")

    with open(logging_conf_file) as conf_file:
        logging.config.dictConfig(yaml.load(conf_file))

    app = connexion.FlaskApp(__name__, specification_dir='')
    app.add_api(
        '../api.yml',
        arguments={'api_local': 'local_value'},
        options={"swagger_ui": False}
    )

    logging.info("API started")

    return app.app
