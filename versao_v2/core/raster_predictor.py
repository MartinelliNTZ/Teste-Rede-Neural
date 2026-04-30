from __future__ import annotations

import math
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np
import psutil
import rasterio
from rasterio.windows import Window

from .hardware_manager import calculate_chunk_lines


def _read_chunk(args: Tuple[str, int, int, int]):
    path_src, row_start, width, n_rows = args
    with rasterio.open(path_src) as src:
        window = Window(0, row_start, width, n_rows)
        data = src.read(window=window)
    return row_start, n_rows, data


class RasterPredictor:
    def __init__(
        self,
        batch_size: int = 4096,
        use_mask: bool = True,
        alpha_threshold: int = 250,
        ram_limit_bytes: Optional[int] = None,
        progress_callback: Optional[Callable[[int, str], None]] = None,
    ):
        self.batch_size = batch_size
        self.use_mask = use_mask
        self.alpha_threshold = alpha_threshold
        self.ram_limit_bytes = ram_limit_bytes
        self.progress_callback = progress_callback

    def _report_progress(self, percent: int, message: str) -> None:
        if self.progress_callback:
            self.progress_callback(percent, message)

    def predict(
        self,
        source_path: Path,
        model,
        n_bands_feature: int,
        output_path: Path,
    ) -> Path:
        source_path = Path(source_path)
        output_path = Path(output_path)
        self._validate_paths(source_path)

        with rasterio.open(str(source_path)) as src:
            height = src.height
            width = src.width
            n_bands_total = src.count
            out_meta = src.meta.copy()

        n_feat_classif = n_bands_total - 1 if self.use_mask else n_bands_total
        if n_feat_classif != n_bands_feature:
            raise ValueError(
                f"Incompatibilidade de bandas: treino usou {n_bands_feature} bandas, "
                f"mas imagem de classificacao possui {n_feat_classif} bandas de feature."
            )

        ram_available = psutil.virtual_memory().available
        ram_to_use = min(ram_available, self.ram_limit_bytes) if self.ram_limit_bytes else ram_available
        chunk_lines = max(256, calculate_chunk_lines(width, n_bands_feature, ram_to_use) // 12)
        num_chunks = math.ceil(height / chunk_lines)

        out_meta.update(
            {
                "driver": "GTiff",
                "count": 1,
                "dtype": "uint8",
                "compress": "lzw",
                "nodata": 255,
                "height": height,
                "width": width,
                "tiled": True,
                "blockxsize": 512,
                "blockysize": 512,
            }
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)

        pixels_ok = 0
        total_pixels = height * width
        tasks = []
        for i in range(num_chunks):
            row_start = i * chunk_lines
            row_end = min(row_start + chunk_lines, height)
            tasks.append((str(source_path), row_start, width, row_end - row_start))

        with rasterio.open(str(output_path), "w", **out_meta) as dst:
            with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
                for chunk_index, result in enumerate(executor.map(_read_chunk, tasks)):
                    row_start, n_rows, chunk_data = result
                    window = Window(0, row_start, width, n_rows)

                    if self.use_mask and n_bands_total > n_bands_feature:
                        mask_band = chunk_data[-1]
                        features = chunk_data[:n_bands_feature]
                    else:
                        mask_band = None
                        features = chunk_data

                    flattened = features.transpose(1, 2, 0).reshape(-1, n_bands_feature).astype(np.float32)
                    valid_mask = np.ones(flattened.shape[0], dtype=bool)
                    if mask_band is not None:
                        valid_mask = mask_band.reshape(-1) >= self.alpha_threshold

                    result_arr = np.full(flattened.shape[0], 255, dtype=np.uint8)
                    num_valid = int(valid_mask.sum())
                    if num_valid > 0:
                        pred_raw = model.predict(flattened[valid_mask], batch_size=self.batch_size, verbose=0)
                        if pred_raw.ndim == 2 and pred_raw.shape[1] == 1:
                            pred_cls = np.round(pred_raw).astype(np.uint8).flatten()
                        else:
                            pred_cls = np.argmax(pred_raw, axis=1).astype(np.uint8)
                        result_arr[valid_mask] = pred_cls

                    dst.write(result_arr.reshape(1, n_rows, width), window=window)
                    pixels_ok += num_valid
                    percent = int((chunk_index + 1) / num_chunks * 100)
                    self._report_progress(percent, f"Chunk {chunk_index + 1}/{num_chunks}")

        with rasterio.open(str(output_path), "r+") as dst:
            overviews = [2, 4, 8, 16, 32, 64]
            dst.build_overviews(overviews, rasterio.enums.Resampling.nearest)
            dst.update_tags(ns="rio_overview", resampling="nearest")

        self._report_progress(100, "Predição concluída")
        return output_path

    def _validate_paths(self, source_path: Path) -> None:
        if not source_path.is_file():
            raise FileNotFoundError(f"Imagem de classificação não encontrada: {source_path}")
