from __future__ import annotations

from pathlib import Path
from tensorflow.keras.models import Model


class Trainer:
    @staticmethod
    def train(
        model: Model,
        X_train,
        Y_train,
        epochs: int = 150,
        batch_size: int = 64,
        validation_split: float = 0.25,
        verbose: int = 1,
    ):
        history = model.fit(
            X_train,
            Y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=verbose,
        )
        return history

    @staticmethod
    def save_model(model: Model, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(path))
