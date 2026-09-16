"""
Closed-form memory / communication estimates (the formulas from the ZeRO paper,
Rajbhandari et al. 2020) so we can check the simulator against theory and then
extrapolate from our 4M-parameter toy to the 7B / 70B / 405B models real labs train.

Mixed-precision Adam, per parameter:
    working weights (bf16)  2 bytes   -> replicated in DP, ZeRO-1, ZeRO-2 ; sharded in ZeRO-3
    gradients (bf16)        2 bytes   -> replicated in DP, ZeRO-1        ; sharded in ZeRO-2/3
    fp32 master weights     4 bytes   -+
    Adam m (fp32)           4 bytes    |-> K = 12 bytes "optimizer state", sharded from ZeRO-1 on
    Adam v (fp32)           4 bytes   -+
"""
from __future__ import annotations

from dataclasses import dataclass

PARAM_BYTES = 2
GRAD_BYTES = 2
OPTIM_BYTES = 12  # fp32 master + m + v


@dataclass
class MemoryEstimate:
    params: float
    grads: float
    optim: float

    @property
    def total(self):
        return self.params + self.grads + self.optim


def model_state_bytes(psi: float, N: int, stage: str) -> MemoryEstimate:
    """Persistent model-state bytes per GPU (activations excluded -- ZeRO does not shard them)."""
    if stage == "DP":
        return MemoryEstimate(PARAM_BYTES * psi, GRAD_BYTES * psi, OPTIM_BYTES * psi)
    if stage == "ZeRO-1":
        return MemoryEstimate(PARAM_BYTES * psi, GRAD_BYTES * psi, OPTIM_BYTES * psi / N)
    if stage == "ZeRO-2":
        return MemoryEstimate(PARAM_BYTES * psi, GRAD_BYTES * psi / N, OPTIM_BYTES * psi / N)
    if stage == "ZeRO-3":
        return MemoryEstimate(PARAM_BYTES * psi / N, GRAD_BYTES * psi / N, OPTIM_BYTES * psi / N)
    raise ValueError(stage)


def comm_volume_bytes(psi: float, N: int, stage: str, wire_bytes: int = 2) -> float:
    """Bytes each rank sends per training step using ring collectives.
    reduce-scatter and all-gather each move (N-1)/N * psi elements per rank."""
    unit = wire_bytes * psi * (N - 1) / N
    return {"DP": 2 * unit, "ZeRO-1": 2 * unit, "ZeRO-2": 2 * unit, "ZeRO-3": 3 * unit}[stage]


def activation_bytes_transformer(batch: int, seq: int, hidden: int, layers: int, bytes_per=2, checkpointing=False) -> float:
    """Very rough transformer activation footprint (~34 * s*b*h per layer w/o checkpointing,
    Korthikanti et al. 2022, ignoring the attention-score term).  With full activation
    checkpointing only the layer inputs (2*s*b*h) are kept."""
    per_layer = (2 if checkpointing else 34) * seq * batch * hidden * bytes_per / 2
    return per_layer * layers


def fmt_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024:
            return f"{n:,.1f} {unit}"
        n /= 1024
    return f"{n:,.1f} PB"
