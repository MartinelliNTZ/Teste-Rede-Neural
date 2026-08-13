#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Debug point buffer conversion"""

import fiona
from shapely.geometry import shape

BUFFER_M = 0.5

shp_path = './1-AETHERIS_CLASSIFIER_/solo.shp'

with fiona.open(shp_path) as f:
    print(f"Geometry type: {f.schema['geometry']}")
    print(f"Total features: {len(f)}")
    print()
    
    features_list = list(f)
    
    for i in range(min(5, len(features_list))):
        feat = features_list[i]
        raw = feat.get("geometry")
        print(f"Feature {i}:")
        print(f"  Raw geom: {raw}")
        
        try:
            g = shape(raw)
            print(f"  After shape(): geom_type={g.geom_type}, is_valid={g.is_valid}, is_empty={g.is_empty}")
            
            g_buffered = g.buffer(BUFFER_M)
            print(f"  After buffer({BUFFER_M}): geom_type={g_buffered.geom_type}, is_valid={g_buffered.is_valid}, is_empty={g_buffered.is_empty}, area={g_buffered.area}")
            
            g_cleaned = g_buffered.buffer(0)
            print(f"  After buffer(0): geom_type={g_cleaned.geom_type}, is_valid={g_cleaned.is_valid}, is_empty={g_cleaned.is_empty}")
            
        except Exception as e:
            print(f"  ERROR: {type(e).__name__}: {e}")
        
        print()
