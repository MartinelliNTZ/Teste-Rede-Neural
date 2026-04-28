#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
VETORIZADOR RASTER INTELIGENTE v1
================================
Fluxo automático:
1. Procura raster .tif/.tiff na mesma pasta do script
2. Aplica filtro 3x3 majority
3. Remove regiões menores que X m²
4. Vetoriza
5. Dissolve por classe
6. Salva GeoPackage + Shapefile

Dependências:
pip install rasterio geopandas shapely scipy scikit-image fiona pyogrio

Uso:
python vetorizar.py
"""

import os
import time
from pathlib import Path
from collections import Counter

import numpy as np
import rasterio
from rasterio.features import shapes
from shapely.geometry import shape
import geopandas as gpd

from scipy.ndimage import generic_filter
from skimage.measure import label


# ==========================================================
# CONFIG
# ==========================================================
MIN_AREA_M2 = 25          # remove manchas menores que isso
USE_8_CONN = True         # conectividade
NODATA_DEFAULT = 255
OUTPUT_NAME = "resultado_vetorizado"

# ==========================================================
# UTIL
# ==========================================================
BASE_DIR = Path(__file__).resolve().parent


def fmt(sec):
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = sec % 60
    return f"{h:02d}:{m:02d}:{s:05.2f}"


def progress(step, total, t0, label):
    done = step / total
    elapsed = time.perf_counter() - t0
    if done > 0:
        eta = elapsed / done - elapsed
    else:
        eta = 0
    bar = "█" * int(done * 30) + "░" * (30 - int(done * 30))
    print(
        f"\r{label:<25} [{bar}] {done*100:6.2f}% | "
        f"Tempo {fmt(elapsed)} | ETA {fmt(eta)}",
        end="",
        flush=True
    )


def find_raster():
    rasters = list(BASE_DIR.glob("*.tif")) + list(BASE_DIR.glob("*.tiff"))
    if not rasters:
        raise FileNotFoundError("Nenhum raster .tif/.tiff encontrado.")
    return rasters[0]


# ==========================================================
# FILTRO MAJORITY
# ==========================================================
def majority_func(values):
    vals = values.astype(np.int32)
    vals = vals[vals != NODATA_DEFAULT]
    if len(vals) == 0:
        return NODATA_DEFAULT
    return Counter(vals).most_common(1)[0][0]


def majority_filter(arr):
    print("\n[1/4] Aplicando filtro majority 3x3")
    t0 = time.perf_counter()
    out = generic_filter(
        arr,
        function=majority_func,
        size=3,
        mode="nearest"
    )
    print(f"\r[1/4] Concluído em {fmt(time.perf_counter()-t0)}")
    return out.astype(arr.dtype)


# ==========================================================
# REMOVE ÁREAS PEQUENAS
# ==========================================================
def remove_small_regions(arr, pixel_area):
    print("\n[2/4] Removendo regiões pequenas")

    classes = np.unique(arr)
    classes = classes[classes != NODATA_DEFAULT]

    out = arr.copy()
    t0 = time.perf_counter()

    for i, cls in enumerate(classes, 1):
        progress(i - 1, len(classes), t0, "Classes processadas")

        mask = arr == cls
        conn = 2 if USE_8_CONN else 1
        lab = label(mask, connectivity=conn)

        ids, counts = np.unique(lab, return_counts=True)

        for rid, cnt in zip(ids, counts):
            if rid == 0:
                continue

            area = cnt * pixel_area
            if area < MIN_AREA_M2:
                out[lab == rid] = NODATA_DEFAULT

        progress(i, len(classes), t0, "Classes processadas")

    print(f"\n[2/4] Concluído em {fmt(time.perf_counter()-t0)}")
    return out


# ==========================================================
# VETORIZAÇÃO
# ==========================================================
def polygonize(arr, transform, crs):
    print("\n[3/4] Vetorizando raster")

    geoms = []
    vals = []

    t0 = time.perf_counter()

    generator = list(shapes(arr, transform=transform))
    total = len(generator)

    for i, (geom, val) in enumerate(generator, 1):
        if val == NODATA_DEFAULT:
            continue

        geoms.append(shape(geom))
        vals.append(int(val))

        if i % 500 == 0 or i == total:
            progress(i, total, t0, "Polígonos")

    print(f"\n[3/4] Concluído em {fmt(time.perf_counter()-t0)}")

    gdf = gpd.GeoDataFrame(
        {"classe": vals},
        geometry=geoms,
        crs=crs
    )
    return gdf


# ==========================================================
# DISSOLVE
# ==========================================================
def dissolve_classes(gdf):
    print("\n[4/4] Dissolvendo por classe")
    t0 = time.perf_counter()

    gdf["area_m2"] = gdf.area
    out = gdf.dissolve(by="classe", as_index=False)
    out["area_m2"] = out.area

    print(f"[4/4] Concluído em {fmt(time.perf_counter()-t0)}")
    return out


# ==========================================================
# MAIN
# ==========================================================
def main():
    global NODATA_DEFAULT

    print("=" * 70)
    print("VETORIZADOR RASTER INTELIGENTE")
    print("=" * 70)

    raster_path = find_raster()
    print(f"\nRaster encontrado: {raster_path.name}")

    t_global = time.perf_counter()

    with rasterio.open(raster_path) as src:
        arr = src.read(1)
        transform = src.transform
        crs = src.crs
        nodata = src.nodata

        if nodata is not None:
            NODATA_DEFAULT = nodata

        pixel_area = abs(src.res[0] * src.res[1])

        print(f"Dimensão     : {src.width:,} x {src.height:,}")
        print(f"Pixel size   : {src.res}")
        print(f"Área pixel   : {pixel_area:.4f} m²")
        print(f"Nodata       : {NODATA_DEFAULT}")
        print(f"Classes      : {np.unique(arr)}")

    # 1
    arr = majority_filter(arr)

    # 2
    arr = remove_small_regions(arr, pixel_area)

    # 3
    gdf = polygonize(arr, transform, crs)

    # 4
    gdf = dissolve_classes(gdf)

    # SAVE
    gpkg = BASE_DIR / f"{OUTPUT_NAME}.gpkg"
    shp = BASE_DIR / f"{OUTPUT_NAME}.shp"

    print("\nSalvando arquivos...")
    gdf.to_file(gpkg, driver="GPKG")
    gdf.to_file(shp)

    total = time.perf_counter() - t_global

    print("\n" + "=" * 70)
    print("PROCESSAMENTO FINALIZADO")
    print("=" * 70)
    print(f"GeoPackage : {gpkg.name}")
    print(f"Shapefile  : {shp.name}")
    print(f"Classes    : {len(gdf)}")
    print(f"Tempo total: {fmt(total)}")
    print("=" * 70)


if __name__ == "__main__":
    main()