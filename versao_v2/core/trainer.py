from __future__ import annotations

import os
import sys

# Suppress AVX CPU instruction set warning - redirect stderr before TensorFlow loads
# This must be done BEFORE importing tensorflow
AVX_WARNING = "To enable the following instructions"
if sys.stderr is not None:
    # Create a filtered stderr that suppresses AVX warnings
    import io
    class _FilteredStderr(io.TextIOBase):
        def __init__(self, original):
            self._original = original
        def write(self, text):
            if AVX_WARNING not in str(text):
                self._original.write(text)
        def flush(self):
            self._original.flush()
        def __getattr__(self, attr):
            return getattr(self._original, attr)
    
    # Apply filter if sys.stderr exists
    try:
        sys.stderr = _FilteredStderr(sys.stderr)
    except Exception:
        pass  # Keep original stderr if filtering fails

from pathlib import Path
from typing import Callable, Optional
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import Callback


class UICallback(Callback):
    """Custom callback that sends training progress to UI console."""
    def __init__(self, logger: Optional[Callable[[str], None]] = None, 
                 total_epochs: int = 150):
        super().__init__()
        self.logger = logger
        self.total_epochs = total_epochs
        
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        acc = logs.get('accuracy', 0)
        loss = logs.get('loss', 0)
        val_acc = logs.get('val_accuracy', 0)
        val_loss = logs.get('val_loss', 0)
        
        if self.logger:
            self.logger(
                f"Epoch {epoch+1}/{self.total_epochs} - "
                f"acc: {acc:.4f} - loss: {loss:.4f} - "
                f"val_acc: {val_acc:.4f} - val_loss: {val_loss:.4f}"
            )


class Trainer:
    @staticmethod
    def train(
        model: Model,
        X_train,
        Y_train,
        epochs: int = 150,
        batch_size: int = 64,
        validation_split: float = 0.25,
        verbose: int = 0,
        logger: Optional[Callable[[str], None]] = None,
    ):
        # Use callback to send progress to UI instead of printing to terminal
        callbacks = []
        if logger:
            callbacks.append(UICallback(logger=logger, total_epochs=epochs))
            
        history = model.fit(
            X_train,
            Y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=verbose,  # Suppress terminal output
            callbacks=callbacks,
        )
        return history

    @staticmethod
    def save_model(model: Model, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(path))
