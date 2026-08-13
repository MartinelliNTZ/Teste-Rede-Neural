#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Test point buffer conversion"""

import fiona
import rasterio
from shapely.geometry import shape, mapping
from shapely.ops import transform as shp_transform
from pyproj import CRS, Transformer
import os

BUFFER_M = 0.5

def reproject_geom(geom, src_crs, dst_crs):
    t = Transformer.from_crs(src_crs, dst_crs, always_xy=True)
    return shp_transform(t.transform, geom)

def load_geoms_test(shp_path, tiff_crs):
    geoms = []
    with fiona.open(shp_path) as f:
        shp_crs = CRS.from_user_input(f.crs)
        reproj  = not shp_crs.equals(tiff_crs, ignore_axis_order=True)
        
        # Detecta tipo de geometria para aplicar buffer se for Point
        is_point = f.schema['geometry'] == 'Point'
        print(f"\n  Geometry type: {f.schema['geometry']}")
        print(f"  Apply buffer: {is_point}")
        print(f"  Total features: {len(f)}")
        
        count = 0
        for i, feat in enumerate(f):
            raw = feat.get("geometry")
            if raw is None:
                continue
            try:
                g = shape(raw)
                
                # Se é ponto, aplica buffer ANTES de limpar (converte para polígono circular)
                if is_point:
                    g = g.buffer(BUFFER_M)
                
                # Depois limpa a geometria
                g = g.buffer(0)
                    
            except Exception as e:
                print(f"    Error in feature {i}: {e}")
                continue
            if g.is_empty or not g.is_valid:
                print(f"    Feature {i}: empty or invalid")
                continue
            if reproj:
                g = reproject_geom(g, shp_crs, tiff_crs)
            geoms.append(g)
            count += 1
            if count <= 3:
                print(f"    Feature {i}: area={g.area:.2f}, type={g.geom_type}, valid={g.is_valid}")
    
    print(f"  Valid geometries: {len(geoms)}")
    return geoms

# Check TIFF
tiff_path = './1-AETHERIS_CLASSIFIER_/CLASSIFICADA.tif'
with rasterio.open(tiff_path) as src:
    tiff_crs = src.crs
    print(f"TIFF CRS: {tiff_crs}")

# Test shapefiles
shapefiles = [
    './1-AETHERIS_CLASSIFIER_/solo.shp',
    './1-AETHERIS_CLASSIFIER_/palhada.shp',
    './1-AETHERIS_CLASSIFIER_/vegetacao.shp'
]

for shp_path in shapefiles:
    print(f"\n{os.path.basename(shp_path)}:")
    if os.path.exists(shp_path):
        geoms = load_geoms_test(shp_path, tiff_crs)
    else:
        print(f"  NOT FOUND")
