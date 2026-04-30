from __future__ import annotations

import numpy as np
from pathlib import Path
from typing import Optional, Tuple

from .raster_source import RasterSource


class FeatureExtractor:
    @staticmethod
    def extract(
        raster: RasterSource,
        geopandas_frame,
        use_mask: bool = True,
        alpha_threshold: int = 250,
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        if geopandas_frame is None or geopandas_frame.empty:
            raise ValueError("GeoDataFrame de amostras está vazio")

        coords = [(float(x), float(y)) for x, y in zip(geopandas_frame.geometry.x, geopandas_frame.geometry.y)]
        valores = raster.sample(coords)

        if valores.ndim != 2:
            raise ValueError("A extração espectral retornou um array com dimensões inválidas")

        n_bands = raster.count
        if use_mask and n_bands > 1:
            n_bands_feature = n_bands - 1
            X = valores[:, :n_bands_feature]
        else:
            n_bands_feature = n_bands
            X = valores

        if np.isnan(X).any():
            raise ValueError("Foram encontrados valores NaN na extração espectral")

        Y = geopandas_frame["id"].to_numpy(dtype=np.int32).reshape(-1, 1)
        return X.astype(np.float32), Y, n_bands_feature
