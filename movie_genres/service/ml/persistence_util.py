"""Helper module for dumping and loading pickles."""

from pathlib import Path
import pickle
from typing import Any


def dump_object(obj: Any, file_path: Path) -> None:
    """Dump object."""
    with file_path.open("wb") as file:
        pickle.dump(obj, file)


def load_object(file_path: Path) -> Any:
    """Load object."""
    with file_path.open("rb") as file:
        return pickle.load(file)
