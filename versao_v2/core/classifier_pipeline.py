from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import geopandas as gpd
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam

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
        self._run_started_at: Optional[datetime] = None
        self._report_dir: Optional[Path] = None

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

    @staticmethod
    def _build_report_paths(now: datetime) -> Tuple[Path, Path, str]:
        stamp = now.strftime("%Y%m%d_%H%M%S")
        base_name = f"report_{stamp}"
        report_root = Path("report")
        report_root.mkdir(parents=True, exist_ok=True)
        html_path = report_root / f"{base_name}.html"
        assets_dir = report_root / base_name
        assets_dir.mkdir(parents=True, exist_ok=True)
        return html_path, assets_dir, base_name

    @staticmethod
    def _format_seconds(total_seconds: float) -> str:
        total = max(int(total_seconds), 0)
        hours = total // 3600
        minutes = (total % 3600) // 60
        seconds = total % 60
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    def _build_html_report(
        self,
        report_path: Path,
        assets_folder_name: str,
        evaluation: EvaluationResult,
        output_path: Path,
        execution_info_path: Optional[Path],
        train_info: Dict[str, object],
        class_info: Dict[str, object],
        shapefiles_info: List[Dict[str, object]],
        started_at: datetime,
        finished_at: datetime,
    ) -> None:
        duration_seconds = (finished_at - started_at).total_seconds()
        duration_hms = self._format_seconds(duration_seconds)
        model_path = self._last_saved_model_path or self.config.existing_model_path

        def _img_tag(path: Path, label: str) -> str:
            if not path.exists():
                return f"<p><b>{label}:</b> arquivo nao encontrado ({path.name})</p>"
            return (
                f"<h3>{label}</h3>"
                f"<img src='{assets_folder_name}/{path.name}' alt='{label}' style='max-width:100%; border:1px solid #3E3E42; border-radius:8px;'/>"
            )

        class_rows = []
        for class_name in self.class_names:
            row = evaluation.report.get(class_name, {})
            if not row:
                continue
            class_rows.append(
                "<tr>"
                f"<td>{class_name}</td>"
                f"<td>{float(row.get('precision', 0.0)):.4f}</td>"
                f"<td>{float(row.get('recall', 0.0)):.4f}</td>"
                f"<td>{float(row.get('f1-score', 0.0)):.4f}</td>"
                f"<td>{int(row.get('support', 0))}</td>"
                "</tr>"
            )
        class_table_html = "\n".join(class_rows) if class_rows else "<tr><td colspan='5'>Sem dados</td></tr>"

        train_pixels = int(train_info.get("width", 0)) * int(train_info.get("height", 0))
        class_pixels = int(class_info.get("width", 0)) * int(class_info.get("height", 0))

        shp_items = []
        for shp in shapefiles_info:
            shp_items.append(
                f"<li>[Classe {shp.get('class_id')}] {shp.get('legend')} - {Path(str(shp.get('path', '-'))).name} "
                f"({shp.get('features', 0)} feicoes)</li>"
            )
        shapefile_html = "".join(shp_items) if shp_items else "<li>Sem shapefiles.</li>"

        html = f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Report Execucao - {started_at.strftime("%Y-%m-%d %H:%M:%S")}</title>
  <style>
    body {{ background:#121212; color:#EAEAEA; font-family:Segoe UI, Arial, sans-serif; margin:24px; }}
    .card {{ background:#1E1E1E; border:1px solid #3E3E42; border-radius:10px; padding:14px; margin-bottom:14px; }}
    h1, h2, h3 {{ color:#D4A853; margin:0 0 10px 0; }}
    p, li {{ color:#D0D0D0; }}
    ul {{ margin:6px 0 0 18px; }}
    .muted {{ color:#A0A0A0; }}
    table {{ width:100%; border-collapse: collapse; }}
    th, td {{ border:1px solid #3E3E42; padding:8px; text-align:left; font-size:13px; }}
    th {{ background:#252526; color:#EAEAEA; }}
  </style>
</head>
<body>
  <h1>Report de Execucao</h1>
  <div class="card">
    <h2>Resumo</h2>
    <p><b>Inicio:</b> {started_at.strftime("%Y-%m-%d %H:%M:%S")}</p>
    <p><b>Fim:</b> {finished_at.strftime("%Y-%m-%d %H:%M:%S")}</p>
    <p><b>Tempo total:</b> {duration_hms} ({duration_seconds:.2f}s)</p>
    <p><b>Acuracia:</b> {evaluation.accuracy:.4f} ({evaluation.accuracy * 100:.2f}%)</p>
  </div>
  <div class="card">
    <h2>Hardware e Pipeline</h2>
    <p><b>Dispositivo:</b> {getattr(self.hardware_info, "device", "-")}</p>
    <p><b>RAM limite:</b> {float(getattr(self.hardware_info, "ram_limit_gb", 0.0)):.2f} GB ({self.config.ram_limit_pct}%)</p>
    <p><b>Acao do modelo:</b> {self.config.model_action}</p>
    <p><b>Camadas ocultas:</b> {self.config.hidden_layers} | <b>Ativacao:</b> {self.config.activation} | <b>Dropout:</b> {self.config.dropout_rate}</p>
    <p><b>Mascara:</b> {self.config.use_mask} (alpha >= {self.config.alpha_threshold})</p>
  </div>
  <div class="card">
    <h2>Arquivos Utilizados</h2>
    <ul>
      <li>Imagem treino: {self.config.training_image}</li>
      <li>Imagem classificacao: {self.config.classification_image}</li>
      <li>Saida classificada (TIF): {output_path}</li>
      <li>Modelo: {model_path if model_path else "Nao aplicavel"}</li>
      <li>Info JSON: {assets_folder_name}/{Path(str(execution_info_path)).name if execution_info_path else "Nao gerado"}</li>
      <li>Relatorio metricas TXT: {assets_folder_name}/{evaluation.report_path.name}</li>
    </ul>
  </div>
  <div class="card">
    <h2>Validacao das Imagens</h2>
    <p><b>Treino:</b> {Path(str(train_info.get("path", "-"))).name} | {train_info.get("width", 0)} x {train_info.get("height", 0)} px | bandas: {train_info.get("bands", 0)} | pixels: {train_pixels:,}</p>
    <p><b>CRS treino:</b> {train_info.get("crs", "-")}</p>
    <p><b>Classificacao:</b> {Path(str(class_info.get("path", "-"))).name} | {class_info.get("width", 0)} x {class_info.get("height", 0)} px | bandas: {class_info.get("bands", 0)} | pixels: {class_pixels:,}</p>
    <p><b>CRS classificacao:</b> {class_info.get("crs", "-")}</p>
  </div>
  <div class="card">
    <h2>Amostras por Classe</h2>
    <ul>{shapefile_html}</ul>
  </div>
  <div class="card">
    <h2>Metricas por Classe</h2>
    <table>
      <thead><tr><th>Classe</th><th>Precisao</th><th>Recall</th><th>F1-Score</th><th>Suporte</th></tr></thead>
      <tbody>
        {class_table_html}
      </tbody>
    </table>
  </div>
  <div class="card">
    <h2>Graficos de Avaliacao</h2>
    {_img_tag(evaluation.confusion_matrix_path, "Matriz de Confusao")}
    {_img_tag(evaluation.plot_loss_path, "Curvas de Loss e Accuracy")}
  </div>
</body>
</html>
"""
        report_path.write_text(html, encoding="utf-8")

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

    def _expected_output_units(self) -> int:
        return 1 if len(self.class_names) == 2 else len(self.class_names)

    def _expected_loss(self) -> str:
        return "binary_crossentropy" if len(self.class_names) == 2 else "sparse_categorical_crossentropy"

    def _load_existing_model(self):
        if not self.config.existing_model_path:
            raise ValueError("Modelo existente nao informado")
        self._log(f"Carregando modelo existente: {self.config.existing_model_path}")
        return load_model(str(self.config.existing_model_path))

    def _validate_model_compatibility(self, n_bands_feature: int) -> None:
        if self.model is None:
            raise ValueError("Modelo nao carregado")

        model_input_shape = getattr(self.model, "input_shape", None)
        if not model_input_shape or len(model_input_shape) < 2:
            raise ValueError("Nao foi possivel validar input_shape do modelo existente")

        model_input_features = model_input_shape[-1]
        if model_input_features != n_bands_feature:
            raise ValueError(
                "Modelo existente incompativel com o raster de treino: "
                f"modelo espera {model_input_features} bandas, mas os dados atuais possuem {n_bands_feature}."
            )

        model_output_shape = getattr(self.model, "output_shape", None)
        if not model_output_shape or len(model_output_shape) < 2:
            raise ValueError("Nao foi possivel validar output_shape do modelo existente")

        model_output_units = model_output_shape[-1]
        expected_output_units = self._expected_output_units()
        if model_output_units != expected_output_units:
            raise ValueError(
                "Modelo existente incompativel com as classes atuais: "
                f"modelo possui saida com {model_output_units} unidade(s), "
                f"mas a configuracao atual requer {expected_output_units}."
            )

    def _ensure_model_compiled_for_training(self) -> None:
        if self.model is None:
            raise ValueError("Modelo nao carregado")

        is_compiled = getattr(self.model, "optimizer", None) is not None
        if is_compiled:
            return

        loss = self._expected_loss()
        self._log(f"Modelo existente sem estado de compilacao. Recompilando com loss={loss}.")
        self.model.compile(loss=loss, optimizer=Adam(), metrics=["accuracy"])

    def execute(self) -> PipelineResult:
        self.config.validate()
        self._run_started_at = datetime.now()
        report_html_path, self._report_dir, report_base_name = self._build_report_paths(self._run_started_at)
        self._log("Iniciando pipeline do classificador")

        self.hardware_info = configure_hardware(self.config.ram_limit_pct)
        self._log(f"Hardware: {self.hardware_info.device} | RAM limite {self.hardware_info.ram_limit_gb:.2f} GB")
        self._log(
            f"Pipeline: treino={self.config.training_image} | classif={self.config.classification_image} | "
            f"saida={self.config.output_path}"
        )
        artifact_paths = self._build_artifact_paths()

        train_raster = RasterSource(self.config.training_image)
        class_raster = RasterSource(self.config.classification_image)
        train_raster.validate()
        class_raster.validate()
        train_info = self._collect_raster_info(train_raster)
        class_info = self._collect_raster_info(class_raster)
        self._log(
            "Validacao imagens: "
            f"Treino {train_info['width']}x{train_info['height']} ({int(train_info['width']) * int(train_info['height']):,} px) | "
            f"Classif {class_info['width']}x{class_info['height']} ({int(class_info['width']) * int(class_info['height']):,} px)"
        )

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

        self._log("Preparando o modelo")
        if self.config.model_action == "Treinar modelo novo":
            self.model = ModelFactory.build(
                input_shape=(split.X_train.shape[1],),
                num_classes=len(self.class_names),
                hidden_layers=self.config.hidden_layers,
                activation=self.config.activation,
                dropout_rate=self.config.dropout_rate,
            )
        else:
            self.model = self._load_existing_model()
            self._validate_model_compatibility(n_bands_feature)
            self._log("Modelo existente validado com sucesso")

        if self.config.model_action != "Usar modelo existente":
            self._ensure_model_compiled_for_training()
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
        shapefiles_info = self._collect_shapefiles_info()

        self._log("Avaliando modelo")
        evaluation = Evaluator.evaluate(
            self.model,
            split.X_test,
            split.Y_test,
            class_names=self.class_names,
            output_dir=self._report_dir,
            history=self.history,
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

        if self._last_saved_info_path and self._last_saved_info_path.exists():
            shutil.copy2(self._last_saved_info_path, self._report_dir / self._last_saved_info_path.name)

        finished_at = datetime.now()
        self._build_html_report(
            report_path=report_html_path,
            assets_folder_name=report_base_name,
            evaluation=evaluation,
            output_path=output_path,
            execution_info_path=self._last_saved_info_path,
            train_info=train_info,
            class_info=class_info,
            shapefiles_info=shapefiles_info,
            started_at=self._run_started_at,
            finished_at=finished_at,
        )
        self._log(f"Report HTML salvo em {report_html_path}")
        self._log(f"Arquivos do report salvos em {self._report_dir} ({report_base_name})")

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
