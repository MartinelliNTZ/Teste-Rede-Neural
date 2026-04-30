from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


class PipelineConfigError(ValueError):
    pass


@dataclass(frozen=True)
class ShapefileEntry:
    path: Path
    class_id: int
    legend: str = ""


@dataclass
class PipelineConfig:
    training_image: Path
    classification_image: Path
    output_path: Path
    shapefiles: List[ShapefileEntry]
    model_action: str
    save_model: bool
    model_path: Path
    existing_model_path: Optional[Path]
    test_size: float
    random_state: int
    epochs: int
    batch_size_train: int
    batch_size_pred: int
    hidden_layers: List[int]
    activation: str
    dropout_rate: float
    use_mask: bool
    alpha_threshold: int
    ram_limit_pct: int

    VALID_ACTIONS = (
        "Treinar modelo novo",
        "Treinar modelo existente",
        "Usar modelo existente",
    )
    VALID_ACTIVATIONS = ("relu", "elu", "tanh", "sigmoid", "linear")

    def __post_init__(self):
        self.training_image = Path(self.training_image)
        self.classification_image = Path(self.classification_image)
        self.output_path = Path(self.output_path)
        self.model_path = Path(self.model_path)
        if self.existing_model_path is not None:
            self.existing_model_path = Path(self.existing_model_path)

        self.validate()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PipelineConfig":
        try:
            shapefiles = []
            for item in data.get("shapefiles", []):
                if isinstance(item, dict):
                    path = item["path"]
                    class_id = item["class_id"]
                    legend = item.get("legend", "")
                elif isinstance(item, (list, tuple)) and len(item) >= 2:
                    path = item[0]
                    class_id = item[1]
                    legend = item[2] if len(item) > 2 else ""
                else:
                    raise PipelineConfigError("Formato de shapefile inválido")
                shapefiles.append(ShapefileEntry(Path(path), int(class_id), str(legend)))

            hidden_layers = data.get("hidden_layers", data.get("camadas", []))
            if isinstance(hidden_layers, str):
                hidden_layers = [int(value) for value in hidden_layers.replace(" ", "").split(",") if value]
            elif isinstance(hidden_layers, (list, tuple)):
                hidden_layers = [int(value) for value in hidden_layers]
            else:
                hidden_layers = []

            existing_model_path = data.get("existing_model_path")
            if existing_model_path is None:
                existing_model_path = data.get("modelo_existente") or data.get("model_path_existing")

            return cls(
                training_image=Path(data["training_image"]),
                classification_image=Path(data["classification_image"]),
                output_path=Path(data["output"]),
                shapefiles=shapefiles,
                model_action=str(data.get("model_action", "Treinar modelo novo")),
                save_model=bool(data.get("save_model", False)),
                model_path=Path(data.get("model_path", "resultado/modelo.keras")),
                existing_model_path=Path(existing_model_path) if existing_model_path else None,
                test_size=float(data.get("test_size", 0.3)),
                random_state=int(data.get("random_state", 42)),
                epochs=int(data.get("epochs", 150)),
                batch_size_train=int(data.get("batch_size_train", data.get("batch_size_treino", 64))),
                batch_size_pred=int(data.get("batch_size_pred", data.get("batch_size_predicao", 4096))),
                hidden_layers=hidden_layers,
                activation=str(data.get("activation", data.get("ativacao", "relu"))),
                dropout_rate=float(data.get("dropout_rate", data.get("dropout", 0.1))),
                use_mask=bool(data.get("use_mask", data.get("usar_mascara", True))),
                alpha_threshold=int(data.get("alpha_threshold", data.get("valor_minimo_alpha", 250))),
                ram_limit_pct=int(data.get("ram_limit_pct", data.get("spin_ram", 70))),
            )
        except KeyError as exc:
            raise PipelineConfigError(f"Chave obrigatória ausente: {exc}") from exc
        except ValueError as exc:
            raise PipelineConfigError(f"Valor inválido na configuração: {exc}") from exc

    def validate(self) -> None:
        if self.model_action not in self.VALID_ACTIONS:
            raise PipelineConfigError(f"Ação de modelo inválida: {self.model_action}")

        if not self.training_image.is_file():
            raise PipelineConfigError(f"Imagem de treino não encontrada: {self.training_image}")
        if not self.classification_image.is_file():
            raise PipelineConfigError(f"Imagem de classificação não encontrada: {self.classification_image}")

        if not self.shapefiles:
            raise PipelineConfigError("Nenhum shapefile de treino foi fornecido")

        duplicates = [class_id for class_id in {entry.class_id for entry in self.shapefiles} if [e for e in self.shapefiles if e.class_id == class_id].__len__() > 1]
        if duplicates:
            raise PipelineConfigError(f"IDs de classe duplicados nos shapefiles: {duplicates}")

        for entry in self.shapefiles:
            if not entry.path.is_file():
                raise PipelineConfigError(f"Shapefile não encontrado: {entry.path}")

        if self.model_action != "Treinar modelo novo" and self.existing_model_path is None:
            raise PipelineConfigError("Modelo existente deve ser informado para a ação escolhida")
        if self.existing_model_path is not None and not self.existing_model_path.is_file():
            raise PipelineConfigError(f"Modelo existente não encontrado: {self.existing_model_path}")

        if self.save_model and not self.model_path:
            raise PipelineConfigError("Caminho de salvamento do modelo deve ser informado")

        if not 0.0 < self.test_size < 1.0:
            raise PipelineConfigError("test_size deve estar entre 0 e 1")

        if self.dropout_rate < 0 or self.dropout_rate >= 1:
            raise PipelineConfigError("dropout_rate deve estar entre 0 e 1")

        if self.activation not in self.VALID_ACTIVATIONS:
            raise PipelineConfigError(f"Ativação inválida: {self.activation}")

        if self.ram_limit_pct < 1 or self.ram_limit_pct > 100:
            raise PipelineConfigError("ram_limit_pct deve estar entre 1 e 100")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "training_image": str(self.training_image),
            "classification_image": str(self.classification_image),
            "output": str(self.output_path),
            "shapefiles": [
                {"path": str(entry.path), "class_id": entry.class_id, "legend": entry.legend}
                for entry in self.shapefiles
            ],
            "model_action": self.model_action,
            "save_model": self.save_model,
            "model_path": str(self.model_path),
            "existing_model_path": str(self.existing_model_path) if self.existing_model_path else None,
            "test_size": self.test_size,
            "random_state": self.random_state,
            "epochs": self.epochs,
            "batch_size_train": self.batch_size_train,
            "batch_size_pred": self.batch_size_pred,
            "hidden_layers": self.hidden_layers,
            "activation": self.activation,
            "dropout_rate": self.dropout_rate,
            "use_mask": self.use_mask,
            "alpha_threshold": self.alpha_threshold,
            "ram_limit_pct": self.ram_limit_pct,
        }

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: Path) -> "PipelineConfig":
        path = Path(path)
        if not path.is_file():
            raise PipelineConfigError(f"Arquivo de configuração não encontrado: {path}")
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        return cls.from_dict(data)
