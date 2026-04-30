from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
import rasterio


class RasterSource:
    def __init__(self, path: Path):
        self.path = Path(path)

    def validate(self) -> None:
        if not self.path.is_file():
            raise FileNotFoundError(f"Raster não encontrado: {self.path}")

    def open(self):
        return rasterio.open(str(self.path))

    def read_window(self, row_start: int, width: int, height: int):
        with self.open() as src:
            window = rasterio.windows.Window(0, row_start, width, height)
            return src.read(window=window)

    def sample(self, coordinates: Iterable[Tuple[float, float]]) -> np.ndarray:
        with self.open() as src:
            samples = list(src.sample(coordinates))
        result = np.asarray(samples)
        if np.ma.isMaskedArray(result):
            result = result.filled(np.nan)
        return result

    @property
    def width(self) -> int:
        with self.open() as src:
            return src.width

    @property
    def height(self) -> int:
        with self.open() as src:
            return src.height

    @property
    def count(self) -> int:
        with self.open() as src:
            return src.count

    @property
    def crs(self):
        with self.open() as src:
            return src.crs

    @property
    def meta(self) -> dict:
        with self.open() as src:
            return src.meta.copy()
