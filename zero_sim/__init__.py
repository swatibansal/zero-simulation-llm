"""zero_sim -- a 32-virtual-GPU sandbox for understanding DP vs ZeRO-1/2/3."""
import os as _os

# Each virtual GPU is a Python thread doing NumPy matmuls.  If BLAS also spawns a
# thread pool per call we get 32 x n_cores threads thrashing -- the step time
# exploded ~50x before this was pinned.  (Real clusters have the same class of
# problem: CPU-side data-loader / NCCL-proxy threads oversubscribing host cores.)
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    _os.environ.setdefault(_v, "1")
try:  # also cover the case where numpy was imported before us
    from threadpoolctl import threadpool_limits as _tpl

    _tpl(limits=1)
except Exception:  # pragma: no cover
    pass

from .analytic import activation_bytes_transformer, comm_volume_bytes, fmt_bytes, model_state_bytes
from .cluster import Cluster, HardwareProfile, MemoryLedger, VirtualGPU
from .model import MLP, make_dataset
from .trainers import TRAINERS, AdamConfig, DataParallelTrainer, StepStats, ZeRO1Trainer, ZeRO2Trainer, ZeRO3Trainer

__all__ = [
    "Cluster", "HardwareProfile", "MemoryLedger", "VirtualGPU", "MLP", "make_dataset",
    "TRAINERS", "AdamConfig", "DataParallelTrainer", "StepStats", "ZeRO1Trainer", "ZeRO2Trainer", "ZeRO3Trainer",
    "model_state_bytes", "comm_volume_bytes", "activation_bytes_transformer", "fmt_bytes",
]
