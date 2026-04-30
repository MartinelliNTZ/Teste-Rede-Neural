from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class SplitResult:
    X_train: Any
    X_test: Any
    Y_train: Any
    Y_test: Any
    class_counts: Dict[int, int]


class DatasetSplitter:
    @staticmethod
    def split(X, Y, test_size: float = 0.3, random_state: int = 42) -> SplitResult:
        if X is None or Y is None:
            raise ValueError("Dados de entrada não encontrados")

        if len(X) != len(Y):
            raise ValueError("X e Y devem ter o mesmo número de amostras")

        y_flat = np.asarray(Y).reshape(-1)
        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, test_size=test_size, random_state=random_state, stratify=y_flat
        )

        class_counts = {int(label): int(np.sum(y_flat == label)) for label in np.unique(y_flat)}
        return SplitResult(
            X_train=X_train,
            X_test=X_test,
            Y_train=Y_train,
            Y_test=Y_test,
            class_counts=class_counts,
        )
