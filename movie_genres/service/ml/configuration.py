"""Training and scoring configuration module."""

from dataclasses import dataclass


@dataclass
class TrainConfiguration():
    """Training configuration class."""

    batch_size: int = 128
    epochs: int = 1
    early_stop: int = 5
    validation_split: float = 0.2


@dataclass
class EmbeddingConfiguration():
    """Word embedding configuration class."""

    max_words: int = 100_000
    max_len: int = 150
    embed_size: int = 300
    embed_file: str = f"data/glove.6B.300d.txt"
