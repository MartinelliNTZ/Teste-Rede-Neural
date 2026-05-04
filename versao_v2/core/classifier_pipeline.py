from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional

import geopandas as gpd
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
    info_path: Optional[Path]
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
        self._last_saved_model_path: Optional[Path] = None
        self._last_saved_info_path: Optional[Path] = None

    def _log(self, message: str) -> None:
        self.logger(message)

    def _progress(self, percent: int, message: str) -> None:
        if self.progress_callback:
            self.progress_callback(percent, message)

    @staticmethod
    def _json_safe(value):
        if isinstance(value, dict):
            return {str(k): ClassifierPipeline._json_safe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [ClassifierPipeline._json_safe(v) for v in value]
        if isinstance(value, Path):
            return str(value)
        if hasattr(value, "item"):
            try:
                return value.item()
            except Exception:
                pass
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return str(value)

    @staticmethod
    def _month_folder_name(now: datetime) -> str:
        month_names = {
            1: "JANEIRO",
            2: "FEVEREIRO",
            3: "MARCO",
            4: "ABRIL",
            5: "MAIO",
            6: "JUNHO",
            7: "JULHO",
            8: "AGOSTO",
            9: "SETEMBRO",
            10: "OUTUBRO",
            11: "NOVEMBRO",
            12: "DEZEMBRO",
        }
        return f"{now.year}-{month_names[now.month]}"

    @staticmethod
    def _with_timestamp_suffix(base_name: str, suffix: str) -> str:
        safe_base = Path(base_name).stem
        return f"{safe_base}_{suffix}"

    def _build_artifact_paths(self) -> Dict[str, Path]:
        now = datetime.now()
        timestamp = now.strftime("%Y%m%d_%H%M%S")
        month_folder = self._month_folder_name(now)
        base_dir = Path("models") / month_folder
        base_dir.mkdir(parents=True, exist_ok=True)

        if self.config.model_action == "Usar modelo existente" and self.config.existing_model_path:
            model_base_name = self.config.existing_model_path.stem
        else:
            model_base_name = self.config.model_path.stem

        final_base_name = self._with_timestamp_suffix(model_base_name, timestamp)
        return {
            "base_dir": base_dir,
            "model": base_dir / f"{final_base_name}.keras",
            "info": base_dir / f"{final_base_name}.info.json",
        }

    def _collect_raster_info(self, raster: RasterSource) -> Dict[str, object]:
        with raster.open() as src:
            transform = src.transform
            return {
                "path": str(raster.path),
                "width": src.width,
                "height": src.height,
                "bands": src.count,
                "dtype": [str(dt) for dt in src.dtypes],
                "crs": str(src.crs) if src.crs else None,
                "nodata": src.nodata,
                "bounds": list(src.bounds),
                "transform": [transform.a, transform.b, transform.c, transform.d, transform.e, transform.f],
                "meta": src.meta.copy(),
            }

    def _collect_shapefiles_info(self) -> List[Dict[str, object]]:
        info: List[Dict[str, object]] = []
        for entry in self.config.shapefiles:
            gdf = gpd.read_file(str(entry.path))
            info.append(
                {
                    "path": str(entry.path),
                    "class_id": entry.class_id,
                    "legend": entry.legend or entry.path.stem,
                    "features": int(len(gdf)),
                    "crs": str(gdf.crs) if gdf.crs else None,
                    "geometry_types": sorted({str(geom_type) for geom_type in gdf.geom_type.dropna().tolist()}),
                    "columns": [str(col) for col in gdf.columns],
                    "bounds": list(gdf.total_bounds) if len(gdf) else None,
                }
            )
        return info

    def _save_execution_info(
        self,
        info_path: Path,
        train_raster: RasterSource,
        class_raster: RasterSource,
        n_bands_feature: int,
    ) -> None:
        payload = {
            "generated_at": datetime.now().isoformat(),
            "model_action": self.config.model_action,
            "saved_model_path": str(self._last_saved_model_path) if self._last_saved_model_path else None,
            "used_existing_model_path": str(self.config.existing_model_path) if self.config.existing_model_path else None,
            "config": self.config.to_dict(),
            "hardware": {
                "device": getattr(self.hardware_info, "device", None),
                "ram_limit_gb": getattr(self.hardware_info, "ram_limit_gb", None),
                "ram_limit_bytes": getattr(self.hardware_info, "ram_limit_bytes", None),
            },
            "features": {
                "bands_used": n_bands_feature,
                "class_names": self.class_names,
            },
            "raster_training": self._collect_raster_info(train_raster),
            "raster_classification": self._collect_raster_info(class_raster),
            "shapefiles": self._collect_shapefiles_info(),
        }
        info_path.parent.mkdir(parents=True, exist_ok=True)
        with info_path.open("w", encoding="utf-8") as handle:
            json.dump(self._json_safe(payload), handle, indent=2, ensure_ascii=False)

    def execute(self) -> PipelineResult:
        self.config.validate()
        self._log("Iniciando pipeline do classificador")

        self.hardware_info = configure_hardware(self.config.ram_limit_pct)
        self._log(f"Hardware: {self.hardware_info.device} | RAM limite {self.hardware_info.ram_limit_gb:.2f} GB")
        artifact_paths = self._build_artifact_paths()

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
                logger=self._log,
            )
            if self.config.save_model:
                self._log(f"Salvando modelo em {artifact_paths['model']}")
                Trainer.save_model(self.model, artifact_paths["model"])
                self._last_saved_model_path = artifact_paths["model"]

        self._save_execution_info(
            info_path=artifact_paths["info"],
            train_raster=train_raster,
            class_raster=class_raster,
            n_bands_feature=n_bands_feature,
        )
        self._last_saved_info_path = artifact_paths["info"]
        self._log(f"Info de execucao salva em {artifact_paths['info']}")

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
            model_path=self._last_saved_model_path,
            info_path=self._last_saved_info_path,
            history=self.history,
            class_names=self.class_names,
        )
