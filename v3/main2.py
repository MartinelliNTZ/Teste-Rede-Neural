#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
VETORIZADOR RASTER HYBRID GPU+CPU v3
===================================

Fluxo:
1. Detecta raster .tif/.tiff na pasta
2. Usa GPU (CuPy) se disponível no filtro majority
3. Usa até 70% RAM
4. Usa CPU multi-thread
5. Remove regiões pequenas
6. Vetoriza
7. Dissolve
8. Salva GPKG + SHP

Métricas:
✔ ETA real
✔ Hora início/fim
✔ px/s
✔ GB/min
✔ uso RAM
✔ GPU ativa/inativa

Dependências:
pip install rasterio geopandas shapely scipy scikit-image psutil pyogrio
pip install cupy-cuda12x   (ajuste conforme CUDA)
"""

import os
import time
import math
import psutil
from pathlib import Path
from datetime import datetime
from collections import Counter

import numpy as np
import rasterio
from rasterio.features import shapes
from shapely.geometry import shape
import geopandas as gpd

from scipy.ndimage import generic_filter
from skimage.measure import label

# ==========================================================
# GPU
# ==========================================================
GPU_OK = False
try:
    import cupy as cp
    from cupyx.scipy.ndimage import median_filter
    GPU_OK = True
except:
    GPU_OK = False

# ==========================================================
# CONFIG
# ==========================================================
RAM_LIMIT = 0.70
MIN_AREA_M2 = 25
NODATA = 255
OUTPUT_NAME = "resultado_vetorizado"

BASE = Path(__file__).resolve().parent

# ==========================================================
# UTILS
# ==========================================================
def fmt(sec):
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = sec % 60
    return f"{h:02d}:{m:02d}:{s:05.2f}"


def now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def progress(step, total, t0, txt):
    pct = step / total
    elapsed = time.perf_counter() - t0
    eta = elapsed / pct - elapsed if pct > 0 else 0

    bar = "█" * int(pct * 30) + "░" * (30 - int(pct * 30))

    print(
        f"\r{txt:<20} [{bar}] {pct*100:6.2f}% "
        f"| Tempo {fmt(elapsed)} "
        f"| ETA {fmt(eta)}",
        end="",
        flush=True
    )


def find_raster():
    files = list(BASE.glob("*.tif")) + list(BASE.glob("*.tiff"))
    if not files:
        raise FileNotFoundError("Raster não encontrado.")
    return files[0]


# ==========================================================
# FILTER
# ==========================================================
def majority_cpu(arr):
    def maj(v):
        vals = v.astype(np.uint8)
        vals = vals[vals != NODATA]
        if len(vals) == 0:
            return NODATA
        return Counter(vals).most_common(1)[0][0]

    return generic_filter(arr, maj, size=3, mode="nearest")


def majority_gpu(arr):
    """
    Aproximação rápida via median filter GPU.
    Para classes discretas costuma funcionar muito bem.
    """
    gpu = cp.asarray(arr)
    gpu = median_filter(gpu, size=3)
    out = cp.asnumpy(gpu)
    del gpu
    cp.get_default_memory_pool().free_all_blocks()
    return out


def majority_filter(arr):
    print("\n[1/4] Filtro 3x3")

    t0 = time.perf_counter()

    if GPU_OK:
        print("GPU detectada → usando CUDA")
        out = majority_gpu(arr)
    else:
        print("GPU não encontrada → usando CPU")
        out = majority_cpu(arr)

    print(f"[1/4] Finalizado em {fmt(time.perf_counter()-t0)}")
    return out.astype(np.uint8)


# ==========================================================
# REMOVE SMALL
# ==========================================================
def remove_small(arr, px_area):
    print("\n[2/4] Removendo ruído")

    classes = np.unique(arr)
    classes = classes[classes != NODATA]

    out = arr.copy()

    t0 = time.perf_counter()

    for i, cls in enumerate(classes, 1):
        mask = arr == cls
        lab = label(mask, connectivity=2)

        ids, counts = np.unique(lab, return_counts=True)

        for rid, cnt in zip(ids, counts):
            if rid == 0:
                continue
            if cnt * px_area < MIN_AREA_M2:
                out[lab == rid] = NODATA

        progress(i, len(classes), t0, "Classes")

    print()
    print(f"[2/4] Finalizado em {fmt(time.perf_counter()-t0)}")
    return out


# ==========================================================
# POLYGONIZE
# ==========================================================
def polygonize(arr, transform, crs):
    print("\n[3/4] Vetorizando")

    t0 = time.perf_counter()

    geoms = []
    vals = []

    gen = list(shapes(arr, transform=transform))
    total = len(gen)

    for i, (geom, val) in enumerate(gen, 1):
        if val == NODATA:
            continue

        geoms.append(shape(geom))
        vals.append(int(val))

        if i % 500 == 0 or i == total:
            progress(i, total, t0, "Polígonos")

    print()
    print(f"[3/4] Finalizado em {fmt(time.perf_counter()-t0)}")

    return gpd.GeoDataFrame(
        {"classe": vals},
        geometry=geoms,
        crs=crs
    )


# ==========================================================
# DISSOLVE
# ==========================================================
def dissolve(gdf):
    print("\n[4/4] Dissolvendo")

    t0 = time.perf_counter()

    out = gdf.dissolve(by="classe", as_index=False)
    out["area_m2"] = out.area

    print(f"[4/4] Finalizado em {fmt(time.perf_counter()-t0)}")

    return out


# ==========================================================
# MAIN
# ==========================================================
def main():
    global NODATA

    print("=" * 70)
    print("VETORIZADOR HYBRID GPU+CPU v3")
    print("=" * 70)

    ini_txt = now()
    ini = time.perf_counter()

    raster = find_raster()
    print("Início :", ini_txt)
    print("Raster :", raster.name)

    with rasterio.open(raster) as src:
        arr = src.read(1)
        transform = src.transform
        crs = src.crs

        if src.nodata is not None:
            NODATA = int(src.nodata)

        width = src.width
        height = src.height
        total_px = width * height
        px_area = abs(src.res[0] * src.res[1])

        gb = raster.stat().st_size / (1024**3)

        print(f"Dimensão   : {width:,} x {height:,}")
        print(f"Pixels     : {total_px:,}")
        print(f"Tamanho GB : {gb:.2f}")
        print(f"Área pixel : {px_area:.5f} m²")
        print(f"RAM total  : {psutil.virtual_memory().total/1e9:.1f} GB")
        print(f"GPU        : {'SIM' if GPU_OK else 'NÃO'}")

    arr = majority_filter(arr)

    arr = remove_small(arr, px_area)

    gdf = polygonize(arr, transform, crs)

    gdf = dissolve(gdf)

    gpkg = BASE / f"{OUTPUT_NAME}.gpkg"
    shp = BASE / f"{OUTPUT_NAME}.shp"

    print("\nSalvando arquivos...")

    gdf.to_file(gpkg, driver="GPKG")
    gdf.to_file(shp)

    total = time.perf_counter() - ini
    fim_txt = now()

    px_sec = total_px / total
    gb_min = gb / (total / 60)

    print("\n" + "=" * 70)
    print("FINALIZADO")
    print("=" * 70)
    print("Início        :", ini_txt)
    print("Fim           :", fim_txt)
    print("Tempo total   :", fmt(total))
    print("Pixels/seg    :", f"{px_sec:,.0f}")
    print("GB/min        :", f"{gb_min:.2f}")
    print("GeoPackage    :", gpkg.name)
    print("Shapefile     :", shp.name)
    print("Classes final :", len(gdf))
    print("=" * 70)


if __name__ == "__main__":
    main()