"""Training controller."""

from movie_genres.api.controller import Controller
from movie_genres.ml import train


class TrainController(Controller):
    """Train controller class."""

    def post(self, body):
        """Train POST route implementation."""
        movies = self.read_csv(body)

        train(movies)

        return None, 201
