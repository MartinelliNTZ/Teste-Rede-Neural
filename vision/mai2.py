# -*- coding: utf-8 -*-
"""
Pipeline de Segmentação Semântica com Raster Vision v0.31
----------------------------------------------------------
Entrada : 1 TIFF + 4 shapefiles de polígonos de classes
Saída   : TIFF classificado + GeoJSON/Shapefile vetorizado por classe

ANTES DE RODAR PELA PRIMEIRA VEZ (ou para re-treinar do zero):
    Remove-Item -Recurse -Force D:\teste\vision\saida\train
    Remove-Item -Recurse -Force D:\teste\vision\saida\predict
"""

import os
import glob
import json
import shutil
import tempfile

import fiona
import numpy as np
import rasterio
from rasterio.features import shapes as rasterio_shapes
from pyproj import CRS, Transformer
from shapely.geometry import shape, mapping
from shapely.ops import transform as shapely_transform, unary_union

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

# Apaga treinos anteriores para forçar re-treino limpo
for _subdir in ("train", "predict"):
    _path = os.path.join(OUT_DIR, _subdir)
    if os.path.isdir(_path):
        print(f"[INFO] Removendo treino anterior: {_path}")
        shutil.rmtree(_path)

CHIP_SZ     = 256
NUM_EPOCHS  = 30
BATCH_SZ    = 8      # RTX 4080 aguenta — mais estabilidade no gradiente
LR          = 3e-4
MAX_WINDOWS = 800    # mais chips dentro do AOI

CLASS_NAMES  = ["background", "palhada", "solo", "floresta"]
CLASS_COLORS = ["black",      "yellow",  "brown", "green",   "red"]

# Pesos inversamente proporcionais à frequência esperada:
# background tende a dominar — peso menor; classes raras — peso maior
CLASS_WEIGHTS = [0.2, 1.0, 1.0, 1.0, 1.0]

SHAPES = {
    "palhada":  os.path.join(DATA_DIR, "palhada.shp"),
    "solo":     os.path.join(DATA_DIR, "solo.shp"),
    "floresta": os.path.join(DATA_DIR, "floresta.shp"),
}

# ============================================================
# TIFF
# ============================================================

tiffs = (glob.glob(os.path.join(DATA_DIR, "*.tif")) +
         glob.glob(os.path.join(DATA_DIR, "*.tiff")))
if not tiffs:
    raise FileNotFoundError(f"Nenhum TIFF encontrado em: {DATA_DIR}")
IMAGE_PATH = tiffs[0]
print(f"[INFO] TIFF encontrado: {IMAGE_PATH}")

class_config = ClassConfig(
    names=CLASS_NAMES,
    colors=CLASS_COLORS,
    null_class="background",
)

# ============================================================
# UTILITÁRIOS CRS
# ============================================================

def get_tiff_crs(tiff_path):
    with rasterio.open(tiff_path) as src:
        return src.crs

def crs_from_fiona(fiona_crs):
    try:
        return CRS.from_user_input(fiona_crs)
    except Exception:
        return CRS.from_string(str(fiona_crs))

def crs_are_equivalent(crs_a, crs_b):
    return crs_a.equals(crs_b, ignore_axis_order=True)

def reproject_geometry(geom, src_crs, dst_crs):
    transformer = Transformer.from_crs(src_crs, dst_crs, always_xy=True)
    return shapely_transform(transformer.transform, geom)

# ============================================================
# GEOJSON BUILDER — labels + AOI
# ============================================================

def build_geojson():
    """
    Gera em saida/:
      labels.geojson  — polígonos com class_id
      aoi.geojson     — união dos polígonos (restringe amostragem de chips)
    """
    tiff_crs = get_tiff_crs(IMAGE_PATH)
    print(f"[INFO] CRS do TIFF: {tiff_crs.to_string()}")

    features  = []
    all_geoms = []
    class_map = [("palhada", 1), ("solo", 2), ("floresta", 3), ("daninhas", 4)]

    for class_name, class_id in class_map:
        shp = SHAPES[class_name]
        if not os.path.exists(shp):
            print(f"[AVISO] Shapefile não encontrado, pulando: {shp}")
            continue

        with fiona.open(shp) as src:
            shp_crs        = crs_from_fiona(src.crs)
            need_reproject = not crs_are_equivalent(shp_crs, tiff_crs)
            if need_reproject:
                print(f"[INFO] Reprojetando '{class_name}'")
            else:
                print(f"[INFO] '{class_name}': mesmo CRS, sem reprojeção.")

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
                all_geoms.append(geom)
                count_ok += 1

            print(f"[INFO] {class_name}: {count_ok} OK, {count_skip} descartadas")

    if not features:
        raise RuntimeError("Nenhuma feature válida encontrada nos shapefiles.")

    labels_path = os.path.join(OUT_DIR, "labels.geojson")
    with open(labels_path, "w", encoding="utf-8") as f:
        json.dump({"type": "FeatureCollection", "features": features}, f)
    print(f"[INFO] labels.geojson: {labels_path} ({len(features)} features)")

    # AOI = união dos polígonos + buffer de meio chip
    with rasterio.open(IMAGE_PATH) as src:
        res = abs(src.transform.a)
    half_chip = CHIP_SZ * res / 2

    aoi_geom = unary_union(all_geoms).buffer(half_chip)
    aoi_path = os.path.join(OUT_DIR, "aoi.geojson")
    with open(aoi_path, "w", encoding="utf-8") as f:
        json.dump({
            "type": "FeatureCollection",
            "features": [{"type": "Feature", "geometry": mapping(aoi_geom),
                          "properties": {}}]
        }, f)
    print(f"[INFO] aoi.geojson: {aoi_path}")

    # Estatística: quantos pixels de cada classe existem nos labels
    _report_class_pixel_counts(tiff_crs)

    return labels_path, aoi_path


def _report_class_pixel_counts(tiff_crs):
    """Lê o TIFF e conta quantos pixels cada shapefile cobre — útil para debug."""
    try:
        from rasterio.features import rasterize
        with rasterio.open(IMAGE_PATH) as src:
            meta   = src.meta
            height = src.height
            width  = src.width
            transform = src.transform

        print("\n[INFO] Cobertura aproximada de cada classe no TIFF:")
        total = height * width
        class_map = [("palhada", 1), ("solo", 2), ("floresta", 3), ("daninhas", 4)]
        for class_name, class_id in class_map:
            shp = SHAPES[class_name]
            if not os.path.exists(shp):
                continue
            with fiona.open(shp) as src_shp:
                shp_crs = crs_from_fiona(src_shp.crs)
                need_reproject = not crs_are_equivalent(shp_crs, tiff_crs)
                geoms = []
                for feat in src_shp:
                    raw = feat.get("geometry")
                    if raw is None:
                        continue
                    try:
                        g = shape(raw).buffer(0)
                    except Exception:
                        continue
                    if g.is_empty or not g.is_valid:
                        continue
                    if need_reproject:
                        g = reproject_geometry(g, shp_crs, tiff_crs)
                    geoms.append(g)

            if not geoms:
                print(f"  {class_name}: 0 pixels")
                continue

            burned = rasterize(
                [(g, 1) for g in geoms],
                out_shape=(height, width),
                transform=transform,
                fill=0,
                dtype=np.uint8,
            )
            n = int(burned.sum())
            pct = 100.0 * n / total
            print(f"  {class_name}: {n:,} pixels  ({pct:.2f}% do TIFF)")
        print(f"  Total pixels no TIFF: {total:,}\n")
    except Exception as e:
        print(f"[AVISO] Não foi possível calcular cobertura: {e}")

# ============================================================
# SCENE
# ============================================================

def build_scene(scene_id, labels_path, aoi_path):
    vector_source = GeoJSONVectorSourceConfig(
        uris=[labels_path],
        transformers=[ClassInferenceTransformerConfig(default_class_id=0)],
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
        aoi_uris=[aoi_path],
    )

# ============================================================
# VETORIZAÇÃO
# ============================================================

def vectorize_predictions():
    pred_root = os.path.join(OUT_DIR, "predict")
    pred_tiff = None

    if os.path.isdir(pred_root):
        for dirpath, _, filenames in os.walk(pred_root):
            for fname in filenames:
                if fname == "labels.tif":
                    pred_tiff = os.path.join(dirpath, fname)
                    break
            if pred_tiff:
                break

    if pred_tiff is None:
        print(f"[AVISO] Nenhum labels.tif encontrado em: {pred_root}")
        if os.path.isdir(pred_root):
            for dirpath, _, filenames in os.walk(pred_root):
                for fname in filenames:
                    print(f"  [encontrado] {os.path.join(dirpath, fname)}")
        return

    print(f"[INFO] Vetorizando: {pred_tiff}")
    vec_dir = os.path.join(OUT_DIR, "vetores")
    os.makedirs(vec_dir, exist_ok=True)

    with rasterio.open(pred_tiff) as src:
        data      = src.read(1).astype(np.uint8)
        transform = src.transform
        crs       = src.crs

    # Estatística das predições
    print("[INFO] Distribuição de pixels preditos:")
    total_px = data.size
    for cid, cname in enumerate(CLASS_NAMES):
        n = int((data == cid).sum())
        pct = 100.0 * n / total_px
        print(f"  {cname}: {n:,} px ({pct:.2f}%)")
    print()

    schema = {
        "geometry": "Polygon",
        "properties": {"class_id": "int", "class_name": "str"},
    }

    for class_id, class_name in enumerate(CLASS_NAMES):
        if class_id == 0:
            continue

        mask = (data == class_id).astype(np.uint8)
        if mask.sum() == 0:
            print(f"[INFO] '{class_name}': nenhum pixel predito, pulando.")
            continue

        polys = [
            shape(geom)
            for geom, val in rasterio_shapes(mask, mask=mask, transform=transform)
            if int(val) == 1
        ]
        if not polys:
            continue

        fiona_features = [
            {"geometry": mapping(p),
             "properties": {"class_id": class_id, "class_name": class_name}}
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

        print(f"[OK] '{class_name}' → {out_json}  ({len(fiona_features)} polígonos)")
        print(f"[OK] '{class_name}' → {out_shp}")

    print(f"\n[INFO] Vetores salvos em: {vec_dir}")

# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    from rastervision.pipeline.runner import InProcessRunner
    from rastervision.pipeline.config import build_config, save_pipeline_config

    labels_path, aoi_path = build_geojson()

    scene_train = build_scene("scene_train", labels_path, aoi_path)
    scene_val   = build_scene("scene_val",   labels_path, aoi_path)

    dataset = DatasetConfig(
        class_config=class_config,
        train_scenes=[scene_train],
        validation_scenes=[scene_val],
    )

    backend = PyTorchSemanticSegmentationConfig(
        model=SemanticSegmentationModelConfig(backbone=Backbone.resnet50),
        solver=SolverConfig(
            lr=LR,
            num_epochs=NUM_EPOCHS,
            batch_sz=BATCH_SZ,
            # Penaliza erro nas classes raras — corrige o viés para background
            class_loss_weights=CLASS_WEIGHTS,
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

    with tempfile.TemporaryDirectory() as tmp_dir:
        config.update()
        config.recursive_validate_config()
        build_config(config.dict())

        cfg_json_uri = config.get_config_uri()
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
    print("  TIFF classificado : saida/predict/scene_val/labels.tif")
    print("  Vetores por classe: saida/vetores/classe_<nome>.shp / .geojson")
    print("\n  Verifique as métricas acima:")
    print("  - Se precision/recall das classes ainda for 0 após o treino,")
    print("    os polígonos de treino são pequenos demais em relação ao chip (256px).")
    print("  - Nesse caso, reduza CHIP_SZ para 64 ou 128.")