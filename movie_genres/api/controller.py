"""Abstract Controller module."""

from io import StringIO, BytesIO
from typing import Any

from connexion import problem
from flask import make_response
import pandas as pd


class Controller():
    """Abstract Controller class."""

    def unsupported_media_type(self, request):
        """Create 415 response."""
        content_type = request.headers.get("Content-Type", "")
        return problem(
            415, "Unsupported Media Type",
            f"Invalid Content-type ({content_type}), expected JSON data")

    def csv_response(self, df: pd.DataFrame) -> Any:
        """Create text/csv response from dataframe."""
        bytes_io = StringIO()
        df.to_csv(bytes_io, index=False)

        response = make_response(bytes_io.getvalue())
        response.headers["Content-type"] = "text/csv"

        return response

    def read_csv(self, body):
        """Parse csv from http body."""
        return pd.read_csv(BytesIO(body))
