from __future__ import annotations

from typing import List, Tuple

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout


class ModelFactory:
    @staticmethod
    def build(
        input_shape: Tuple[int, ...],
        num_classes: int,
        hidden_layers: List[int],
        activation: str = "relu",
        dropout_rate: float = 0.0,
    ):
        model = Sequential()
        if not hidden_layers:
            raise ValueError("A arquitetura da rede deve conter ao menos uma camada oculta")

        for index, units in enumerate(hidden_layers):
            if index == 0:
                model.add(Dense(units, activation=activation, input_shape=input_shape))
            else:
                model.add(Dense(units, activation=activation))
            if dropout_rate > 0:
                model.add(Dropout(dropout_rate))

        if num_classes == 2:
            model.add(Dense(1, activation="sigmoid"))
            loss = "binary_crossentropy"
        else:
            model.add(Dense(num_classes, activation="softmax"))
            loss = "sparse_categorical_crossentropy"

        model.compile(loss=loss, optimizer="adam", metrics=["accuracy"])
        return model
