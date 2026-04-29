"""
titanbit.utils.metrics
~~~~~~~~~~~~~~~~~~~~~~
Compute and GPU metrics for training observability.

Tracks:
    - Model Flops Utilisation (MFU)
    - Tokens per second (throughput)
    - VRAM usage and fragmentation
    - GPU temperature and power draw
    - Disk I/O throughput (data pipeline health)
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Optional

import torch

logger = logging.getLogger(__name__)


@dataclass
class GPUMetrics:
    """Snapshot of GPU state."""

    vram_used_gb: float = 0.0
    vram_total_gb: float = 0.0
    vram_utilisation: float = 0.0
    vram_fragmentation: float = 0.0
    gpu_utilisation: float = 0.0
    temperature_c: float = 0.0
    power_w: float = 0.0
    clock_mhz: float = 0.0


def get_gpu_metrics(device_idx: int = 0) -> GPUMetrics:
    """Collect current GPU metrics."""
    if not torch.cuda.is_available():
        return GPUMetrics()

    props = torch.cuda.get_device_properties(device_idx)
    mem_allocated = torch.cuda.memory_allocated(device_idx)
    mem_reserved = torch.cuda.memory_reserved(device_idx)
    mem_total = props.total_mem

    # Fragmentation: how much reserved memory is unused
    frag = 1.0 - (mem_allocated / max(mem_reserved, 1))

    metrics = GPUMetrics(
        vram_used_gb=mem_allocated / (1024**3),
        vram_total_gb=mem_total / (1024**3),
        vram_utilisation=mem_allocated / mem_total,
        vram_fragmentation=frag,
    )

    # Try nvidia-smi metrics via pynvml
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_idx)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
        power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # mW → W
        clock = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM)

        metrics.gpu_utilisation = util.gpu / 100.0
        metrics.temperature_c = temp
        metrics.power_w = power
        metrics.clock_mhz = clock
    except Exception:
        pass  # pynvml not available

    return metrics


class ThroughputTracker:
    """
    Track training throughput with a rolling window.

    Provides stable estimates of tokens/sec that smooth out
    variance from evaluation pauses and checkpoint saves.
    """

    def __init__(self, window_size: int = 50) -> None:
        self._timestamps: deque[float] = deque(maxlen=window_size)
        self._token_counts: deque[int] = deque(maxlen=window_size)
        self._total_tokens = 0
        self._start_time = time.monotonic()

    def update(self, num_tokens: int) -> None:
        self._timestamps.append(time.monotonic())
        self._token_counts.append(num_tokens)
        self._total_tokens += num_tokens

    @property
    def tokens_per_second(self) -> float:
        """Rolling average tokens/sec."""
        if len(self._timestamps) < 2:
            return 0.0
        dt = self._timestamps[-1] - self._timestamps[0]
        total = sum(self._token_counts) - self._token_counts[0]
        return total / max(dt, 1e-6)

    @property
    def overall_tokens_per_second(self) -> float:
        """Overall average tokens/sec since start."""
        elapsed = time.monotonic() - self._start_time
        return self._total_tokens / max(elapsed, 1e-6)

    @property
    def total_tokens(self) -> int:
        return self._total_tokens

    @property
    def elapsed_hours(self) -> float:
        return (time.monotonic() - self._start_time) / 3600

    def eta_hours(self, target_tokens: int) -> float:
        """Estimated time to reach target token count."""
        remaining = target_tokens - self._total_tokens
        if remaining <= 0:
            return 0.0
        tps = self.tokens_per_second
        if tps <= 0:
            return float("inf")
        return (remaining / tps) / 3600


def compute_mfu(
    model_params: int,
    batch_size: int,
    seq_len: int,
    dt: float,
    gpu_flops_peak: float = 119.5e12,
) -> float:
    """
    Compute Model Flops Utilisation.

    MFU = achieved_flops / peak_flops

    For a transformer:
        FLOPs per step ≈ 6 × N × B × T
        (2 for matmul per token, ×3 for forward + backward)

    Parameters
    ----------
    model_params   : non-embedding parameter count
    batch_size     : micro batch size
    seq_len        : sequence length
    dt             : time per step (seconds)
    gpu_flops_peak : theoretical peak FLOPS (L20 BF16 = ~119.5 TFLOPS)
    """
    flops_per_step = 6 * model_params * batch_size * seq_len
    flops_achieved = flops_per_step / max(dt, 1e-9)
    return flops_achieved / gpu_flops_peak


def format_metrics_table(
    step: int,
    loss: float,
    lr: float,
    throughput: ThroughputTracker,
    mfu: float,
    gpu: GPUMetrics,
) -> str:
    """Format a compact metrics line for logging."""
    return (
        f"step={step:>7d} │ "
        f"loss={loss:.4f} │ "
        f"lr={lr:.2e} │ "
        f"tok/s={throughput.tokens_per_second:>8.0f} │ "
        f"MFU={mfu*100:>5.1f}% │ "
        f"VRAM={gpu.vram_used_gb:.1f}/{gpu.vram_total_gb:.1f}GB │ "
        f"T={gpu.temperature_c:.0f}°C"
    )
