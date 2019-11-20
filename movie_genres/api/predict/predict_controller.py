"""Prediction controller."""

from dataclasses import dataclass

import connexion
from dataclasses_json import dataclass_json
import pandas as pd

from movie_genres.api.controller import Controller
from movie_genres.ml import predict


@dataclass_json
@dataclass
class MoviePredict():
    """Movie json http body."""

    synopsis: str


class PredictController(Controller):
    """Prediction controller class."""

    def post(self, body):
        """Prediction POST route implementation."""
        request = connexion.request

        contentType = request.headers["Content-Type"]

        body_parser = {
            "application/json": self.__read_post_json,
            "text/csv": self.read_csv,
        }.get(contentType)

        if body_parser is None:
            return self.unsupported_media_type(request)

        movies = body_parser(body)

        predictions = predict(movies)

        return self.csv_response(predictions)

    def __read_post_json(body):
        movie = MoviePredict.from_json(body)

        return pd.DataFrame(
            {
                "movie_id": [None],
                "synopsis": [movie.synopsis]
            }
        )
