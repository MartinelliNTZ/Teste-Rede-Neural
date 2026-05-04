# -*- coding: utf-8 -*-

import os
import glob
import json
import fiona

from rastervision.core.backend import BackendConfig
from rastervision.pytorch_backend import PyTorchSemanticSegmentationConfig
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

# =============================
# CONFIG
# =============================

ROOT = os.path.abspath(".")
DATA_DIR = os.path.join(ROOT, "dados")
OUT_DIR = os.path.join(ROOT, "saida")

os.makedirs(OUT_DIR, exist_ok=True)

# pega tiff automaticamente
tiffs = glob.glob(os.path.join(DATA_DIR, "*.tif")) + glob.glob(
    os.path.join(DATA_DIR, "*.tiff")
)

if not tiffs:
    raise FileNotFoundError(f"Nenhum arquivo .tif/.tiff encontrado em: {DATA_DIR}")

IMAGE_PATH = tiffs[0]

# shapefiles
SHAPES = {
    "palhada": os.path.join(DATA_DIR, "palhada.shp"),
    "solo": os.path.join(DATA_DIR, "solo.shp"),
    "floresta": os.path.join(DATA_DIR, "floresta.shp"),
    "daninhas": os.path.join(DATA_DIR, "daninhas.shp"),
}

# =============================
# CLASSES (SEM background!)
# =============================

CLASS_NAMES = ["palhada", "solo", "floresta", "daninhas"]

class_config = ClassConfig(names=CLASS_NAMES)

# =============================
# 🔥 FUNÇÃO QUE UNE SHP → GEOJSON
# =============================


from shapely.geometry import shape, mapping


def build_merged_geojson():
    features = []

    for class_id, class_name in enumerate(CLASS_NAMES):
        shp_path = SHAPES[class_name]

        if not os.path.exists(shp_path):
            continue

        with fiona.open(shp_path) as src:
            for feat in src:
                geom = feat["geometry"]

                # 🔥 CONVERSÃO CORRETA
                geom_json = mapping(shape(geom))

                features.append(
                    {
                        "type": "Feature",
                        "geometry": geom_json,
                        "properties": {"class_id": class_id},
                    }
                )

    geojson = {"type": "FeatureCollection", "features": features}

    out_path = os.path.join(OUT_DIR, "labels.geojson")

    with open(out_path, "w") as f:
        json.dump(geojson, f)

    return out_path


# =============================
# LABEL SOURCE (SIMPLES E CORRETO)
# =============================


def build_label_source():
    geojson_path = build_merged_geojson()

    vector_source = GeoJSONVectorSourceConfig(uris=[geojson_path])

    return SemanticSegmentationLabelSourceConfig(
        raster_source=RasterizedSourceConfig(
            vector_source=vector_source,
            rasterizer_config=RasterizerConfig(background_class_id=0),
        )
    )


# =============================
# SCENE
# =============================

scene = SceneConfig(
    id="scene_1",
    raster_source=RasterioSourceConfig(uris=[IMAGE_PATH]),
    label_source=build_label_source(),
)

dataset = DatasetConfig(
    class_config=class_config,  # ✅ AGORA OBRIGATÓRIO AQUI
    train_scenes=[scene],
    validation_scenes=[scene],
)

# =============================
# BACKEND
# =============================

from rastervision.pytorch_backend import (
    PyTorchSemanticSegmentationConfig,
    SemanticSegmentationModelConfig,
    SolverConfig,
    DataConfig
)

backend = PyTorchSemanticSegmentationConfig(
    model=SemanticSegmentationModelConfig(
        backbone='resnet50',   # pode trocar depois
        pretrained=True
    ),
    solver=SolverConfig(
        lr=1e-4,
        num_epochs=10
    ),
    data=DataConfig(
        train_batch_size=4,
        eval_batch_size=4
    )
)

# =============================
# PIPELINE
# =============================
config = SemanticSegmentationConfig(
    root_uri=OUT_DIR,
    dataset=dataset,
    backend=backend,
    train_chip_sz=256,
    predict_chip_sz=256,
)

# =============================
# RUN
# =============================

if __name__ == "__main__":
    pipeline = config.build()
    pipeline.run()
