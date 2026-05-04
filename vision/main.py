# -*- coding: utf-8 -*-

from rastervision.pipeline.rv_config import rv_config_ as rv_config
from rastervision.core.backend import BackendConfig
from rastervision.pytorch_backend import PyTorchSemanticSegmentationConfig
from rastervision.core.data import (
    ClassConfig,
    DatasetConfig,
    SceneConfig,
    RasterioSourceConfig,
    GeoJSONVectorSourceConfig,
    SemanticSegmentationLabelSourceConfig,
    RasterizedSourceConfig
)
from rastervision.core.rv_pipeline import SemanticSegmentationConfig

import os

# =============================
# CONFIGURAÇÕES
# =============================

ROOT = os.path.abspath('.')
DATA_DIR = os.path.join(ROOT, 'dados')
OUT_DIR = os.path.join(ROOT, 'saida')

IMAGE_PATH = os.path.join(DATA_DIR, 'ImagemTreino.tiff')

# shapefiles
SHAPES = {
    'palhada': os.path.join(DATA_DIR, 'palhada.shp'),
    'solo': os.path.join(DATA_DIR, 'solo.shp'),
    'floresta': os.path.join(DATA_DIR, 'floresta.shp'),
    'daninhas': os.path.join(DATA_DIR, 'daninhas.shp')
}

# =============================
# CLASSES
# =============================

class_config = ClassConfig(
    names=['palhada', 'solo', 'floresta', 'daninhas']
)

# =============================
# FUNÇÃO PRA CRIAR LABEL SOURCE
# =============================

def build_label_source():
    vector_sources = []

    for class_id, (class_name, shp_path) in enumerate(SHAPES.items()):
        vector_sources.append(
            GeoJSONVectorSourceConfig(
                uri=shp_path,
                default_class_id=class_id
            )
        )

    return SemanticSegmentationLabelSourceConfig(
        raster_source=RasterizedSourceConfig(
            vector_sources=vector_sources,
            rasterizer_config={
                'background_class_id': 0
            }
        )
    )

# =============================
# SCENE
# =============================

scene = SceneConfig(
    id='scene_1',
    raster_source=RasterioSourceConfig(
        uris=[IMAGE_PATH]
    ),
    label_source=build_label_source()
)

dataset = DatasetConfig(
    train_scenes=[scene],
    validation_scenes=[scene]  # simples (pode melhorar depois)
)

# =============================
# BACKEND (PYTORCH)
# =============================

backend = PyTorchSemanticSegmentationConfig(
    train_batch_size=4,
    eval_batch_size=4,
    num_epochs=10,
    lr=1e-4
)

# =============================
# PIPELINE
# =============================

config = SemanticSegmentationConfig(
    root_uri=OUT_DIR,
    dataset=dataset,
    backend=backend,
    class_config=class_config,
    train_chip_sz=256,
    predict_chip_sz=256
)

# =============================
# EXECUÇÃO
# =============================

if __name__ == '__main__':
    pipeline = config.build()
    pipeline.run()