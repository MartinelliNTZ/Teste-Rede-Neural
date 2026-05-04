# -*- coding: utf-8 -*-

import os
import glob
import json
import fiona

from shapely.geometry import shape, mapping

from rastervision.core.rv_pipeline import (
    SemanticSegmentationConfig,
    SemanticSegmentationPredictOptions,
)
from rastervision.core.data import (
    ClassConfig,
    DatasetConfig,
    SceneConfig,
    RasterioSourceConfig,
    GeoJSONVectorSourceConfig,
    SemanticSegmentationLabelSourceConfig,
    RasterizedSourceConfig,
    RasterizerConfig,
    ClassInferenceTransformerConfig,
)
from rastervision.pytorch_backend import PyTorchSemanticSegmentationConfig
from rastervision.pytorch_learner import (
    SemanticSegmentationModelConfig,
    SemanticSegmentationGeoDataConfig,
    WindowSamplingConfig,
    WindowSamplingMethod,
    SolverConfig,
    Backbone,
)

# =============================
# CONFIG
# =============================

ROOT     = os.path.abspath(".")
DATA_DIR = os.path.join(ROOT, "dados")
OUT_DIR  = os.path.join(ROOT, "saida")

os.makedirs(OUT_DIR, exist_ok=True)

CHIP_SZ     = 256
NUM_EPOCHS  = 10
BATCH_SZ    = 4
LR          = 1e-4
MAX_WINDOWS = 200

# =============================
# TIFF
# =============================

tiffs = (glob.glob(os.path.join(DATA_DIR, "*.tif")) +
         glob.glob(os.path.join(DATA_DIR, "*.tiff")))

if not tiffs:
    raise FileNotFoundError(f"Nenhum TIFF encontrado em: {DATA_DIR}")

IMAGE_PATH = tiffs[0]
print(f"[INFO] TIFF encontrado: {IMAGE_PATH}")

# =============================
# CLASSES
# =============================

CLASS_NAMES = ["background", "palhada", "solo", "floresta", "daninhas"]
# background DEVE ser índice 0 — é o que o RasterizerConfig espera como fundo

SHAPES = {
    "palhada":  os.path.join(DATA_DIR, "palhada.shp"),
    "solo":     os.path.join(DATA_DIR, "solo.shp"),
    "floresta": os.path.join(DATA_DIR, "floresta.shp"),
    "daninhas": os.path.join(DATA_DIR, "daninhas.shp"),
}

class_config = ClassConfig(
    names=CLASS_NAMES,
    colors=["black", "yellow", "brown", "green", "red"],
    null_class="background",
)

# =============================
# GEOJSON BUILDER
# =============================

def build_geojson() -> str:
    features = []

    # class_id 0 = background (não tem shapefile, é o fundo)
    for class_name, class_id in [("palhada", 1), ("solo", 2),
                                   ("floresta", 3), ("daninhas", 4)]:
        shp = SHAPES[class_name]
        if not os.path.exists(shp):
            print(f"[AVISO] Shapefile não encontrado, pulando: {shp}")
            continue

        with fiona.open(shp) as src:
            for feat in src:
                geom = mapping(shape(feat["geometry"]).buffer(0))
                features.append({
                    "type": "Feature",
                    "geometry": geom,
                    "properties": {"class_id": class_id},
                })

    out_path = os.path.join(OUT_DIR, "labels.geojson")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"type": "FeatureCollection", "features": features}, f)

    print(f"[INFO] GeoJSON gerado: {out_path} ({len(features)} features)")
    return out_path

# =============================
# SCENE
# =============================

def build_scene(scene_id: str) -> SceneConfig:
    geojson_path = build_geojson()

    # No v0.31: uris= (lista), transformers= (lista de VectorTransformerConfig)
    # ignore_crs_field e default_class_id foram REMOVIDOS da API
    # Como nosso GeoJSON já tem "class_id" em cada feature,
    # o ClassInferenceTransformerConfig lê esse campo diretamente.
    vector_source = GeoJSONVectorSourceConfig(
        uris=[geojson_path],
        transformers=[
            ClassInferenceTransformerConfig(default_class_id=0),
        ],
    )

    label_source = SemanticSegmentationLabelSourceConfig(
        raster_source=RasterizedSourceConfig(
            vector_source=vector_source,
            rasterizer_config=RasterizerConfig(background_class_id=0),
        )
    )

    return SceneConfig(
        id=scene_id,
        raster_source=RasterioSourceConfig(uris=[IMAGE_PATH]),
        label_source=label_source,
    )

# =============================
# DATASET
# =============================

scene_train = build_scene("scene_train")
scene_val   = build_scene("scene_val")

dataset = DatasetConfig(
    class_config=class_config,
    train_scenes=[scene_train],
    validation_scenes=[scene_val],
)

# =============================
# BACKEND
# =============================

backend = PyTorchSemanticSegmentationConfig(
    model=SemanticSegmentationModelConfig(
        backbone=Backbone.resnet50,
    ),
    solver=SolverConfig(
        lr=LR,
        num_epochs=NUM_EPOCHS,
        batch_sz=BATCH_SZ,
    ),
    data=SemanticSegmentationGeoDataConfig(
        scene_dataset=dataset,
        sampling=WindowSamplingConfig(
            method=WindowSamplingMethod.random,
            size=CHIP_SZ,
            size_lims=(CHIP_SZ, CHIP_SZ + 1),
            max_windows=MAX_WINDOWS,
        ),
        num_workers=0,
    ),
)

# =============================
# PIPELINE
# =============================

config = SemanticSegmentationConfig(
    root_uri=OUT_DIR,
    dataset=dataset,
    backend=backend,
    predict_options=SemanticSegmentationPredictOptions(chip_sz=CHIP_SZ),
)

# =============================
# RUN
# =============================

if __name__ == "__main__":
    from rastervision.pipeline.runner import LocalRunner
    runner = LocalRunner()
    runner.run(config, pipeline_run_config=None, stages=["train"])