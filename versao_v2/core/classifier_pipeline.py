from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional

from tensorflow.keras.models import load_model

from .dataset_splitter import DatasetSplitter
from .evaluator import Evaluator, EvaluationResult
from .feature_extractor import FeatureExtractor
from .hardware_manager import configure_hardware
from .model_factory import ModelFactory
from .pipeline_config import PipelineConfig
from .raster_predictor import RasterPredictor
from .raster_source import RasterSource
from .shapefile_dataset import ShapefileDataset
from .trainer import Trainer


@dataclass(frozen=True)
class PipelineResult:
    hardware_info: object
    evaluation: EvaluationResult
    output_path: Path
    model_path: Optional[Path]
    history: Optional[object]
    class_names: List[str]


class ClassifierPipeline:
    def __init__(
        self,
        config: PipelineConfig,
        logger: Optional[Callable[[str], None]] = None,
        progress_callback: Optional[Callable[[int, str], None]] = None,
    ):
        self.config = config
        self.logger = logger or print
        self.progress_callback = progress_callback
        self.hardware_info = None
        self.model = None
        self.history = None
        self.class_names = []

    def _log(self, message: str) -> None:
        self.logger(message)

    def _progress(self, percent: int, message: str) -> None:
        if self.progress_callback:
            self.progress_callback(percent, message)

    def execute(self) -> PipelineResult:
        self.config.validate()
        self._log("Iniciando pipeline do classificador")

        self.hardware_info = configure_hardware(self.config.ram_limit_pct)
        self._log(f"Hardware: {self.hardware_info.device} | RAM limite {self.hardware_info.ram_limit_gb:.2f} GB")

        train_raster = RasterSource(self.config.training_image)
        class_raster = RasterSource(self.config.classification_image)
        train_raster.validate()
        class_raster.validate()

        self._log("Carregando shapefiles de amostra")
        shapefile_dataset = ShapefileDataset(self.config.shapefiles)
        sample_gdf = shapefile_dataset.load(train_raster.crs)
        self.class_names = [
            shapefile_dataset.get_class_names().get(cls, f"Classe {cls}")
            for cls in sorted({entry.class_id for entry in self.config.shapefiles})
        ]

        self._log("Extraindo valores espectrais")
        X, Y, n_bands_feature = FeatureExtractor.extract(
            train_raster,
            sample_gdf,
            use_mask=self.config.use_mask,
            alpha_threshold=self.config.alpha_threshold,
        )

        split = DatasetSplitter.split(
            X,
            Y,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
        )

        self._log("Construindo o modelo")
        if self.config.model_action == "Treinar modelo novo":
            self.model = ModelFactory.build(
                input_shape=(split.X_train.shape[1],),
                num_classes=len(self.class_names),
                hidden_layers=self.config.hidden_layers,
                activation=self.config.activation,
                dropout_rate=self.config.dropout_rate,
            )
        else:
            self._log("Carregando modelo existente")
            self.model = load_model(str(self.config.existing_model_path))

        if self.config.model_action != "Usar modelo existente":
            self._log("Treinando modelo")
            self.history = Trainer.train(
                self.model,
                split.X_train,
                split.Y_train,
                epochs=self.config.epochs,
                batch_size=self.config.batch_size_train,
            )
            if self.config.save_model:
                self._log(f"Salvando modelo em {self.config.model_path}")
                Trainer.save_model(self.model, self.config.model_path)

        self._log("Avaliando modelo")
        evaluation = Evaluator.evaluate(
            self.model,
            split.X_test,
            split.Y_test,
            class_names=self.class_names,
            output_dir=Path("resultado"),
        )

        self._log("Classificando imagem completa")
        predictor = RasterPredictor(
            batch_size=self.config.batch_size_pred,
            use_mask=self.config.use_mask,
            alpha_threshold=self.config.alpha_threshold,
            ram_limit_bytes=self.hardware_info.ram_limit_bytes,
            progress_callback=self._progress,
        )
        output_path = predictor.predict(
            self.config.classification_image,
            self.model,
            n_bands_feature,
            self.config.output_path,
        )

        self._progress(100, "Pipeline concluído")
        return PipelineResult(
            hardware_info=self.hardware_info,
            evaluation=evaluation,
            output_path=output_path,
            model_path=self.config.model_path if self.config.save_model else None,
            history=self.history,
            class_names=self.class_names,
        )
