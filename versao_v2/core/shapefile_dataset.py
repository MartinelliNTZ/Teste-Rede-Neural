from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import geopandas as gpd
import pandas as pd
from geopandas import GeoDataFrame

from .pipeline_config import ShapefileEntry


class ShapefileDataset:
    def __init__(self, entries: List[ShapefileEntry]):
        self.entries = entries

    def validate(self) -> None:
        if not self.entries:
            raise ValueError("Nenhum shapefile foi informado")
        for entry in self.entries:
            if not entry.path.is_file():
                raise FileNotFoundError(f"Shapefile não encontrado: {entry.path}")

    def load(self, target_crs: Optional[object] = None) -> GeoDataFrame:
        self.validate()
        frames = []
        for entry in self.entries:
            gdf = gpd.read_file(str(entry.path))
            if target_crs is not None and gdf.crs != target_crs:
                gdf = gdf.to_crs(target_crs)
            gdf = gdf.copy()
            gdf["id"] = entry.class_id
            frames.append(gdf)
        if not frames:
            raise ValueError("Nenhum shapefile foi carregado")
        combined = GeoDataFrame(pd.concat(frames, axis=0, ignore_index=True))
        combined.crs = frames[0].crs
        return combined

    def get_class_names(self) -> Dict[int, str]:
        return {entry.class_id: entry.legend or entry.path.stem for entry in self.entries}
