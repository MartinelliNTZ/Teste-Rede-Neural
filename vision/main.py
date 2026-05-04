# -*- coding: utf-8 -*-

import os
import glob
import json
import fiona

from shapely.geometry import shape, mapping

from rastervision.core.data import (
    ClassConfig,
    DatasetConfig,
    SceneConfig,
    RasterioSourceConfig,
    GeoJSONVectorSourceConfig,
    SemanticSegmentationLabelSourceConfig,
    RasterizedSourceConfig,
    RasterizerConfig,
)
from rastervision.core.rv_pipeline import SemanticSegmentationConfig
from rastervision.pytorch_backend import PyTorchSemanticSegmentationConfig

# =============================
# CONFIG
# =============================

ROOT = os.path.abspath(".")
DATA_DIR = os.path.join(ROOT, "dados")
OUT_DIR = os.path.join(ROOT, "saida")

os.makedirs(OUT_DIR, exist_ok=True)

# =============================
# TIFF
# =============================

tiffs = glob.glob(os.path.join(DATA_DIR, "*.tif")) + \
        glob.glob(os.path.join(DATA_DIR, "*.tiff"))

if not tiffs:
    raise FileNotFoundError(f"Nenhum TIFF encontrado em: {DATA_DIR}")

IMAGE_PATH = tiffs[0]

# =============================
# SHAPES
# =============================

SHAPES = {
    "palhada": os.path.join(DATA_DIR, "palhada.shp"),
    "solo": os.path.join(DATA_DIR, "solo.shp"),
    "floresta": os.path.join(DATA_DIR, "floresta.shp"),
    "daninhas": os.path.join(DATA_DIR, "daninhas.shp"),
}

CLASS_NAMES = ["palhada", "solo", "floresta", "daninhas"]

class_config = ClassConfig(names=CLASS_NAMES)

# =============================
# GEOJSON BUILDER
# =============================

def build_geojson():
    features = []

    for class_id, class_name in enumerate(CLASS_NAMES):
        shp = SHAPES[class_name]

        if not os.path.exists(shp):
            continue

        with fiona.open(shp) as src:
            for feat in src:
                geom = mapping(shape(feat["geometry"]).buffer(0))

                features.append({
                    "type": "Feature",
                    "geometry": geom,
                    "properties": {"class_id": class_id},
                })

    out = os.path.join(OUT_DIR, "labels.geojson")

    with open(out, "w") as f:
        json.dump({"type": "FeatureCollection", "features": features}, f)

    return out

# =============================
# LABEL SOURCE
# =============================

def build_label_source():
    geojson = build_geojson()

    return SemanticSegmentationLabelSourceConfig(
        raster_source=RasterizedSourceConfig(
            vector_source=GeoJSONVectorSourceConfig(uris=[geojson]),
            rasterizer_config=RasterizerConfig(background_class_id=0),
        )
    )

# =============================
# SCENE
# =============================

scene = SceneConfig(
    id="scene",
    raster_source=RasterioSourceConfig(uris=[IMAGE_PATH]),
    label_source=build_label_source(),
)

dataset = DatasetConfig(
    class_config=class_config,
    train_scenes=[scene],
    validation_scenes=[scene],
)

# =============================
# BACKEND (VERSÃO ESTÁVEL)
# =============================

backend = PyTorchSemanticSegmentationConfig()

# 👉 overrides simples e seguros
backend.train_chip_sz = 256
backend.predict_chip_sz = 256
backend.num_epochs = 10
backend.train_batch_size = 4
backend.eval_batch_size = 4

# =============================
# PIPELINE
# =============================

config = SemanticSegmentationConfig(
    root_uri=OUT_DIR,
    dataset=dataset,
    backend=backend,
)

# =============================
# RUN
# =============================

if __name__ == "__main__":
    pipeline = config.build()
    pipeline.run()