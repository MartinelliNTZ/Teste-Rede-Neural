# -*- coding: utf-8 -*-
"""
Pipeline de Segmentação Semântica com Raster Vision v0.31
----------------------------------------------------------
Entrada : 1 TIFF + 4 shapefiles de polígonos de classes
Saída   : TIFF classificado + GeoJSON/Shapefile vetorizado por classe

Dependências (além do rastervision):
    pip install pyproj fiona shapely rasterio
"""

import os
import glob
import json
import tempfile

import fiona
import numpy as np
import rasterio
from rasterio.features import shapes as rasterio_shapes
from pyproj import CRS, Transformer
from shapely.geometry import shape, mapping
from shapely.ops import transform as shapely_transform

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

# ============================================================
# CONFIG
# ============================================================

ROOT     = os.path.abspath(".")
DATA_DIR = os.path.join(ROOT, "dados")
OUT_DIR  = os.path.join(ROOT, "saida")

os.makedirs(OUT_DIR, exist_ok=True)

CHIP_SZ     = 256
NUM_EPOCHS  = 10
BATCH_SZ    = 4
LR          = 1e-4
MAX_WINDOWS = 200

# ============================================================
# TIFF — pega o primeiro encontrado na pasta dados/
# ============================================================

tiffs = (glob.glob(os.path.join(DATA_DIR, "*.tif")) +
         glob.glob(os.path.join(DATA_DIR, "*.tiff")))

if not tiffs:
    raise FileNotFoundError(f"Nenhum TIFF encontrado em: {DATA_DIR}")

IMAGE_PATH = tiffs[0]
print(f"[INFO] TIFF encontrado: {IMAGE_PATH}")

# ============================================================
# CLASSES  (background = índice 0, obrigatório)
# ============================================================

CLASS_NAMES  = ["background", "palhada", "solo", "floresta", "daninhas"]
CLASS_COLORS = ["black",      "yellow",  "brown", "green",   "red"]

SHAPES = {
    "palhada":  os.path.join(DATA_DIR, "palhada.shp"),
    "solo":     os.path.join(DATA_DIR, "solo.shp"),
    "floresta": os.path.join(DATA_DIR, "floresta.shp"),
    "daninhas": os.path.join(DATA_DIR, "daninhas.shp"),
}

class_config = ClassConfig(
    names=CLASS_NAMES,
    colors=CLASS_COLORS,
    null_class="background",
)

# ============================================================
# UTILITÁRIOS CRS
# ============================================================

def get_tiff_crs(tiff_path: str) -> CRS:
    with rasterio.open(tiff_path) as src:
        return src.crs


def crs_from_fiona(fiona_crs) -> CRS:
    try:
        return CRS.from_user_input(fiona_crs)
    except Exception:
        return CRS.from_string(str(fiona_crs))


def crs_are_equivalent(crs_a: CRS, crs_b: CRS) -> bool:
    return crs_a.equals(crs_b, ignore_axis_order=True)


def reproject_geometry(geom, src_crs: CRS, dst_crs: CRS):
    transformer = Transformer.from_crs(src_crs, dst_crs, always_xy=True)
    return shapely_transform(transformer.transform, geom)

# ============================================================
# GEOJSON BUILDER
# ============================================================

def build_geojson() -> str:
    tiff_crs = get_tiff_crs(IMAGE_PATH)
    print(f"[INFO] CRS do TIFF: {tiff_crs.to_string()}")

    features = []
    class_map = [("palhada", 1), ("solo", 2), ("floresta", 3), ("daninhas", 4)]

    for class_name, class_id in class_map:
        shp = SHAPES[class_name]
        if not os.path.exists(shp):
            print(f"[AVISO] Shapefile não encontrado, pulando: {shp}")
            continue

        with fiona.open(shp) as src:
            shp_crs = crs_from_fiona(src.crs)
            need_reproject = not crs_are_equivalent(shp_crs, tiff_crs)

            if need_reproject:
                print(f"[INFO] Reprojetando '{class_name}': "
                      f"{shp_crs.to_string()} → {tiff_crs.to_string()}")
            else:
                print(f"[INFO] '{class_name}': mesmo CRS do TIFF, sem reprojeção.")

            count_ok = count_skip = 0
            for feat in src:
                raw_geom = feat.get("geometry")
                if raw_geom is None:
                    count_skip += 1
                    continue

                try:
                    geom = shape(raw_geom).buffer(0)
                except Exception:
                    count_skip += 1
                    continue

                if geom.is_empty or not geom.is_valid:
                    count_skip += 1
                    continue

                if need_reproject:
                    geom = reproject_geometry(geom, shp_crs, tiff_crs)

                features.append({
                    "type": "Feature",
                    "geometry": mapping(geom),
                    "properties": {"class_id": class_id},
                })
                count_ok += 1

            print(f"[INFO] {class_name}: {count_ok} features OK, "
                  f"{count_skip} descartadas")

    if not features:
        raise RuntimeError(
            "Nenhuma feature válida encontrada nos shapefiles.\n"
            "Verifique se os arquivos .shp possuem geometrias."
        )

    out_path = os.path.join(OUT_DIR, "labels.geojson")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"type": "FeatureCollection", "features": features}, f)

    print(f"[INFO] GeoJSON gerado: {out_path} ({len(features)} features)")
    return out_path

# ============================================================
# SCENE
# ============================================================

def build_scene(scene_id: str, geojson_path: str) -> SceneConfig:
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

# ============================================================
# PÓS-PROCESSAMENTO — vetorização com rasterio + fiona
# ============================================================

def vectorize_predictions():
    pred_tiff = os.path.join(OUT_DIR, "predict", "scene_train", "labels.tif")

    if not os.path.exists(pred_tiff):
        print(f"[AVISO] TIFF de predição não encontrado: {pred_tiff}")
        return

    vec_dir = os.path.join(OUT_DIR, "vetores")
    os.makedirs(vec_dir, exist_ok=True)

    with rasterio.open(pred_tiff) as src:
        data      = src.read(1).astype(np.uint8)
        transform = src.transform
        crs       = src.crs

    print(f"[INFO] Vetorizando: {pred_tiff}  |  CRS: {crs.to_string()}")

    schema = {
        "geometry": "Polygon",
        "properties": {"class_id": "int", "class_name": "str"},
    }

    for class_id, class_name in enumerate(CLASS_NAMES):
        if class_id == 0:
            continue

        mask = (data == class_id).astype(np.uint8)
        if mask.sum() == 0:
            print(f"[INFO] Classe '{class_name}': nenhum pixel predito, pulando.")
            continue

        polys = [
            shape(geom)
            for geom, val in rasterio_shapes(mask, mask=mask, transform=transform)
            if int(val) == 1
        ]

        if not polys:
            continue

        fiona_features = [
            {
                "geometry": mapping(p),
                "properties": {"class_id": class_id, "class_name": class_name},
            }
            for p in polys
        ]

        out_json = os.path.join(vec_dir, f"classe_{class_name}.geojson")
        with open(out_json, "w", encoding="utf-8") as fp:
            json.dump({
                "type": "FeatureCollection",
                "crs": {"type": "name", "properties": {"name": crs.to_string()}},
                "features": [{"type": "Feature", **f} for f in fiona_features],
            }, fp)

        out_shp = os.path.join(vec_dir, f"classe_{class_name}.shp")
        with fiona.open(out_shp, "w",
                        driver="ESRI Shapefile",
                        crs=crs.to_wkt(),
                        schema=schema) as dst:
            dst.writerecords(fiona_features)

        print(f"[OK] '{class_name}' → {out_json}")
        print(f"[OK] '{class_name}' → {out_shp}")

    print(f"\n[INFO] Vetores salvos em: {vec_dir}")


# ============================================================
# RUN  —  API correta para Raster Vision v0.31
# ============================================================

if __name__ == "__main__":
    from rastervision.pipeline.runner import InProcessRunner
    from rastervision.pipeline.config import build_config, save_pipeline_config

    # 1. Constrói GeoJSON uma única vez
    geojson_path = build_geojson()

    # 2. Constrói as cenas reutilizando o mesmo GeoJSON
    scene_train = build_scene("scene_train", geojson_path)
    scene_val   = build_scene("scene_val",   geojson_path)

    dataset = DatasetConfig(
        class_config=class_config,
        train_scenes=[scene_train],
        validation_scenes=[scene_val],
    )

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

    config = SemanticSegmentationConfig(
        root_uri=OUT_DIR,
        dataset=dataset,
        backend=backend,
        predict_options=SemanticSegmentationPredictOptions(chip_sz=CHIP_SZ),
    )

    # 3. Serializa o config em JSON (exigido pelo runner)
    #    e constrói o objeto Pipeline dentro de um tmp_dir
    with tempfile.TemporaryDirectory() as tmp_dir:
        config.update()
        config.recursive_validate_config()
        build_config(config.dict())          # valida campos pós-update

        cfg_json_uri = config.get_config_uri()   # <OUT_DIR>/pipeline-config.json
        save_pipeline_config(config, cfg_json_uri)

        pipeline = config.build(tmp_dir)
        runner   = InProcessRunner()

        print("\n=== ETAPA 1: TREINO ===")
        runner.run(cfg_json_uri, pipeline, ["train"])

        print("\n=== ETAPA 2: PREDIÇÃO (TIFF) ===")
        runner.run(cfg_json_uri, pipeline, ["predict"])

    print("\n=== ETAPA 3: VETORIZAÇÃO ===")
    vectorize_predictions()

    print("\n=== PIPELINE CONCLUÍDO ===")
    print(f"Saídas em: {OUT_DIR}")
    print("  TIFF classificado : saida/predict/scene_train/labels.tif")
    print("  Vetores por classe: saida/vetores/classe_<nome>.shp / .geojson")