from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List

import psutil
import tensorflow as tf


@dataclass
class HardwareInfo:
    ram_total_gb: float
    ram_limit_pct: float
    ram_limit_gb: float
    ram_limit_bytes: int
    device: str
    gpu_count: int
    gpu_names: List[str]
    cpu_threads: int


def configure_hardware(ram_limit_pct: float = 70.0) -> HardwareInfo:
    mem = psutil.virtual_memory()
    ram_total_gb = mem.total / (1024 ** 3)
    ram_limit_gb = ram_total_gb * (ram_limit_pct / 100)
    ram_limit_bytes = int(ram_limit_gb * (1024 ** 3))

    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        gpu_names = []
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
                gpu_names.append(gpu.name)
            device = "GPU"
            gpu_count = len(gpus)
        except RuntimeError:
            device = "CPU"
            gpu_count = 0
            gpu_names = []
    else:
        device = "CPU"
        gpu_count = 0
        gpu_names = []

    cpu_count = os.cpu_count() or 4
    tf.config.threading.set_intra_op_parallelism_threads(cpu_count)
    tf.config.threading.set_inter_op_parallelism_threads(cpu_count)

    return HardwareInfo(
        ram_total_gb=ram_total_gb,
        ram_limit_pct=ram_limit_pct,
        ram_limit_gb=ram_limit_gb,
        ram_limit_bytes=ram_limit_bytes,
        device=device,
        gpu_count=gpu_count,
        gpu_names=gpu_names,
        cpu_threads=cpu_count,
    )


def calculate_chunk_lines(width: int, n_bands: int, ram_bytes: int, factor: float = 0.35) -> int:
    bytes_per_line = width * n_bands * 8
    if bytes_per_line <= 0:
        return 1
    return max(1, int(ram_bytes * factor / bytes_per_line))
