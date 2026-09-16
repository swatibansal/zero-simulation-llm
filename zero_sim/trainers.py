"""
Four ways to train the same model on the same 32 virtual GPUs.

                params      grads       optimizer (master + Adam m,v)
  DP            replicated  replicated  replicated          -> 16 * Psi bytes per GPU
  ZeRO-1        replicated  replicated  SHARDED             -> (4 + 12/N) * Psi
  ZeRO-2        replicated  SHARDED     SHARDED             -> (2 + 14/N) * Psi
  ZeRO-3        SHARDED     SHARDED     SHARDED             -> 16/N * Psi   (+ one layer gathered)

All four must produce *identical* weights after every step -- sharding is an
implementation detail of *where bytes live*, not of the math.  `tests/` asserts this.

Communication per step (per rank, in units of Psi = parameter count):
  DP      all-reduce(grads)                 = reduce-scatter + all-gather   = 2 Psi
  ZeRO-1  reduce-scatter(grads) + all-gather(params)                        = 2 Psi
  ZeRO-2  reduce-scatter(grads, per bucket) + all-gather(params)            = 2 Psi
  ZeRO-3  all-gather(params, fwd) + all-gather(params, bwd) + reduce-scatter(grads) = 3 Psi
(each collective really moves (N-1)/N of that, which the ring implementation counts exactly)
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

from .cluster import Cluster, VirtualGPU
from .model import HIGH, LOW, MLP


# --------------------------------------------------------------------------------------
# Adam (the optimizer state that ZeRO-1 exists to shard)
# --------------------------------------------------------------------------------------


@dataclass
class AdamConfig:
    lr: float = 2e-3
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8


def adam_update(master: np.ndarray, m: np.ndarray, v: np.ndarray, grad_low: np.ndarray, t: int, cfg: AdamConfig) -> np.ndarray:
    """In-place Adam on fp32 master weights; returns the new low-precision working copy."""
    g = grad_low.astype(HIGH)
    m *= cfg.beta1
    m += (1 - cfg.beta1) * g
    v *= cfg.beta2
    v += (1 - cfg.beta2) * g * g
    mhat = m / (1 - cfg.beta1**t)
    vhat = v / (1 - cfg.beta2**t)
    master -= cfg.lr * mhat / (np.sqrt(vhat) + cfg.eps)
    return master.astype(LOW)


# --------------------------------------------------------------------------------------
# Base trainer
# --------------------------------------------------------------------------------------


@dataclass
class StepStats:
    step: int
    loss: float
    wall_seconds: float
    comm_bytes_per_rank: int
    comm_sim_seconds: float
    compute_sim_seconds: float
    peak_bytes_per_gpu: int
    peak_by_cat: Dict[str, int] = field(default_factory=dict)


class BaseTrainer:
    stage = "base"

    def __init__(self, cluster: Cluster, model: MLP, adam: AdamConfig | None = None, seed: int = 0):
        self.c = cluster
        self.model = model
        self.N = cluster.N
        self.adam = adam or AdamConfig()
        self.seed = seed
        self.t = 0
        self.L = model.n_layers

    # ---- helpers shared by all stages ------------------------------------------------
    def _chunk(self, gpu: VirtualGPU, flat: np.ndarray, l: int) -> np.ndarray:
        ch = self.model.layers[l].chunk
        return flat[gpu.rank * ch:(gpu.rank + 1) * ch]

    def _init_full_params(self) -> List[np.ndarray]:
        """Rank 0 initialises, then broadcast -- every rank ends with identical fp32 weights."""
        rng = np.random.default_rng(self.seed)
        full = []
        for l in range(self.L):
            root = self.model.init_layer(l, rng)
            full.append(self.c.broadcast([root if r == 0 else np.empty_like(root) for r in range(self.N)])[0])
        return full

    def _forward_layer(self, gpu: VirtualGPU, l: int, flat_low: np.ndarray, x: np.ndarray):
        out, cache = self.model.layer_forward(l, flat_low, x)
        gpu.flops += self.model.layer_flops(l, len(x))
        act_bytes = cache[0].nbytes + (cache[1].nbytes if cache[1] is not None else 0)
        gpu.mem.alloc("activations", act_bytes)
        gpu.store.setdefault("_acts", {})[l] = (cache, act_bytes)
        return out

    def _backward_layer(self, gpu: VirtualGPU, l: int, flat_low: np.ndarray, dout: np.ndarray):
        cache, act_bytes = gpu.store["_acts"].pop(l)
        dx, g = self.model.layer_backward(l, flat_low, cache, dout)
        gpu.flops += self.model.layer_flops(l, len(dout), backward=True)
        gpu.mem.free("activations", act_bytes)  # activation no longer needed once its grad is done
        return dx, g.astype(LOW)

    def _loss_and_dlogits(self, gpu: VirtualGPU, logits, y):
        # scale by 1/N so that SUM over ranks == gradient of the global-batch mean loss
        loss, dl = self.model.softmax_xent(logits, y, scale=1.0 / self.N)
        gpu.store["_loss"] = loss
        return dl

    def _split(self, X, y):
        mb = len(X) // self.N
        return [(X[r * mb:(r + 1) * mb], y[r * mb:(r + 1) * mb]) for r in range(self.N)]

    # ---- one optimisation step with bookkeeping ---------------------------------------
    def step(self, X: np.ndarray, y: np.ndarray) -> StepStats:
        self.t += 1
        self.c.comm.step = self.t
        for g in self.c.gpus:
            g.mem.reset_peaks()
            g.flops = 0
            g.compute_seconds = 0.0
        t0 = time.perf_counter()
        self._step(self._split(X, y))
        wall = time.perf_counter() - t0
        loss = float(np.mean([g.store["_loss"] for g in self.c.gpus]))
        peak_cat = {c: max(g.mem.peak_by_cat[c] for g in self.c.gpus) for c in self.c.gpus[0].mem.peak_by_cat}
        return StepStats(
            step=self.t,
            loss=loss,
            wall_seconds=wall,
            comm_bytes_per_rank=self.c.comm.total_bytes_per_rank(self.t),
            comm_sim_seconds=self.c.comm.total_sim_seconds(self.t),
            compute_sim_seconds=self.c.modelled_compute_seconds(),
            peak_bytes_per_gpu=self.c.peak_bytes_per_gpu(),
            peak_by_cat=peak_cat,
        )

    def _step(self, shards):  # implemented per stage
        raise NotImplementedError

    def full_params(self) -> List[np.ndarray]:
        """Assemble the full low-precision weights from rank 0's view (for equivalence tests)."""
        raise NotImplementedError


# --------------------------------------------------------------------------------------
# Plain data parallel  (everything replicated)
# --------------------------------------------------------------------------------------


class DataParallelTrainer(BaseTrainer):
    stage = "DP"

    def setup(self):
        full = self._init_full_params()
        for gpu in self.c.gpus:
            for l, f in enumerate(full):
                gpu.put(f"p{l}", f.astype(LOW), "params")  # bf16 working weights
                gpu.put(f"master{l}", f.copy(), "master")  # fp32 master
                gpu.put(f"m{l}", np.zeros_like(f), "optim")  # Adam first moment
                gpu.put(f"v{l}", np.zeros_like(f), "optim")  # Adam second moment
        self.c.snapshot("init")
        self.c.reset()

    def _fwd_bwd(self, gpu: VirtualGPU, x, y):
        h = x
        for l in range(self.L):
            h = self._forward_layer(gpu, l, gpu.store[f"p{l}"], h)
        d = self._loss_and_dlogits(gpu, h, y)
        for l in reversed(range(self.L)):
            d, g = self._backward_layer(gpu, l, gpu.store[f"p{l}"], d)
            gpu.put(f"g{l}", g, "grads")  # full local gradient stays resident

    def _step(self, shards):
        self.c.snapshot("step start")
        self.c.comm.phase = "fwd+bwd"
        self.c.parallel(lambda gpu: self._fwd_bwd(gpu, *shards[gpu.rank]))
        self.c.snapshot("after backward")
        self.c.comm.phase = "grad sync"
        for l in range(self.L):  # one all-reduce per layer ("bucket")
            reduced = self.c.all_reduce([g.store[f"g{l}"] for g in self.c.gpus])
            for gpu, r in zip(self.c.gpus, reduced):
                gpu.store[f"g{l}"] = r.astype(LOW)
        self.c.snapshot("after all-reduce")

        def optim(gpu: VirtualGPU):
            for l in range(self.L):
                new = adam_update(gpu.store[f"master{l}"], gpu.store[f"m{l}"], gpu.store[f"v{l}"], gpu.store[f"g{l}"], self.t, self.adam)
                gpu.store[f"p{l}"][...] = new
                gpu.drop(f"g{l}")

        self.c.parallel(optim)
        self.c.snapshot("after optimizer")

    def full_params(self):
        return [self.c.gpus[0].store[f"p{l}"] for l in range(self.L)]


# --------------------------------------------------------------------------------------
# ZeRO-1: shard the optimizer state
# --------------------------------------------------------------------------------------


class ZeRO1Trainer(DataParallelTrainer):
    stage = "ZeRO-1"

    def setup(self):
        full = self._init_full_params()
        for gpu in self.c.gpus:
            for l, f in enumerate(full):
                gpu.put(f"p{l}", f.astype(LOW), "params")  # still a full replica
                ch = self._chunk(gpu, f, l)
                gpu.put(f"master{l}", ch.copy(), "master")  # only MY 1/N of the fp32 master
                gpu.put(f"m{l}", np.zeros_like(ch), "optim")
                gpu.put(f"v{l}", np.zeros_like(ch), "optim")
        self.c.snapshot("init")
        self.c.reset()

    def _reduce_scatter_full_grads(self):
        """Full gradients are resident (like DP) but each rank only needs the reduced chunk it owns."""
        for l in range(self.L):
            shards = self.c.reduce_scatter([g.store[f"g{l}"] for g in self.c.gpus])
            for gpu, s in zip(self.c.gpus, shards):
                gpu.drop(f"g{l}")
                gpu.put(f"gs{l}", s.astype(LOW), "grads")

    def _sharded_optim_and_allgather(self):
        def optim(gpu: VirtualGPU):
            for l in range(self.L):
                new = adam_update(gpu.store[f"master{l}"], gpu.store[f"m{l}"], gpu.store[f"v{l}"], gpu.store[f"gs{l}"], self.t, self.adam)
                gpu.store[f"pnew{l}"] = new  # my updated 1/N chunk (bf16); size-neutral, overwrites into p below
                gpu.drop(f"gs{l}")

        self.c.parallel(optim)
        self.c.comm.phase = "param all-gather"
        for l in range(self.L):
            full = self.c.all_gather([g.store.pop(f"pnew{l}") for g in self.c.gpus])
            for gpu, f in zip(self.c.gpus, full):
                gpu.store[f"p{l}"][...] = f

    def _step(self, shards):
        self.c.snapshot("step start")
        self.c.comm.phase = "fwd+bwd"
        self.c.parallel(lambda gpu: self._fwd_bwd(gpu, *shards[gpu.rank]))
        self.c.snapshot("after backward")
        self.c.comm.phase = "grad reduce-scatter"
        self._reduce_scatter_full_grads()
        self.c.snapshot("after reduce-scatter")
        self._sharded_optim_and_allgather()
        self.c.snapshot("after optimizer + all-gather")


# --------------------------------------------------------------------------------------
# ZeRO-2: shard optimizer state + gradients
# --------------------------------------------------------------------------------------


class ZeRO2Trainer(ZeRO1Trainer):
    stage = "ZeRO-2"

    def _step(self, shards):
        self.c.snapshot("step start")
        self.c.comm.phase = "fwd"

        def fwd(gpu: VirtualGPU):
            x, y = shards[gpu.rank]
            h = x
            for l in range(self.L):
                h = self._forward_layer(gpu, l, gpu.store[f"p{l}"], h)
            gpu.store["_d"] = self._loss_and_dlogits(gpu, h, y)

        self.c.parallel(fwd)
        self.c.snapshot("after forward")
        # Backward layer by layer.  The moment a layer's gradient "bucket" exists we
        # reduce-scatter it and free it, so the full gradient is never resident at once.
        for l in reversed(range(self.L)):
            self.c.comm.phase = f"bwd L{l}"

            def bwd(gpu: VirtualGPU, l=l):
                d, g = self._backward_layer(gpu, l, gpu.store[f"p{l}"], gpu.store["_d"])
                gpu.store["_d"] = d
                gpu.put(f"gbucket{l}", g, "grads")  # transient full-size bucket for ONE layer

            self.c.parallel(bwd)
            self.c.snapshot(f"bwd L{l}: bucket ready")
            shards_l = self.c.reduce_scatter([g.store[f"gbucket{l}"] for g in self.c.gpus])
            for gpu, s in zip(self.c.gpus, shards_l):
                gpu.drop(f"gbucket{l}")
                gpu.put(f"gs{l}", s.astype(LOW), "grads")  # persistent: only my 1/N
            self.c.snapshot(f"bwd L{l}: reduced+freed")
        self._sharded_optim_and_allgather()
        self.c.snapshot("after optimizer + all-gather")


# --------------------------------------------------------------------------------------
# ZeRO-3: shard everything, gather parameters just-in-time
# --------------------------------------------------------------------------------------


class ZeRO3Trainer(ZeRO1Trainer):
    stage = "ZeRO-3"

    def setup(self):
        full = self._init_full_params()
        for gpu in self.c.gpus:
            for l, f in enumerate(full):
                ch = self._chunk(gpu, f, l)
                gpu.put(f"p{l}", ch.astype(LOW), "params")  # only MY 1/N of the bf16 weights
                gpu.put(f"master{l}", ch.copy(), "master")
                gpu.put(f"m{l}", np.zeros_like(ch), "optim")
                gpu.put(f"v{l}", np.zeros_like(ch), "optim")
        self.c.snapshot("init")
        self.c.reset()

    def _gather_layer(self, l: int):
        full = self.c.all_gather([g.store[f"p{l}"] for g in self.c.gpus])
        for gpu, f in zip(self.c.gpus, full):
            gpu.put(f"pfull{l}", f, "temp")  # transient: the whole layer, materialised just-in-time

    def _release_layer(self, l: int):
        for gpu in self.c.gpus:
            gpu.drop(f"pfull{l}")

    def _step(self, shards):
        self.c.snapshot("step start")
        for gpu in self.c.gpus:
            gpu.store["_h"] = shards[gpu.rank][0]
        for l in range(self.L):
            self.c.comm.phase = f"fwd L{l}"
            self._gather_layer(l)
            self.c.snapshot(f"fwd L{l}: gathered")
            self.c.parallel(lambda gpu, l=l: gpu.store.__setitem__("_h", self._forward_layer(gpu, l, gpu.store[f"pfull{l}"], gpu.store["_h"])))
            self._release_layer(l)
            self.c.snapshot(f"fwd L{l}: released")
        self.c.parallel(lambda gpu: gpu.store.__setitem__("_d", self._loss_and_dlogits(gpu, gpu.store.pop("_h"), shards[gpu.rank][1])))
        for l in reversed(range(self.L)):
            self.c.comm.phase = f"bwd L{l}"
            self._gather_layer(l)  # parameters are needed again for dX = dZ @ W^T

            def bwd(gpu: VirtualGPU, l=l):
                d, g = self._backward_layer(gpu, l, gpu.store[f"pfull{l}"], gpu.store["_d"])
                gpu.store["_d"] = d
                gpu.put(f"gbucket{l}", g, "grads")

            self.c.parallel(bwd)
            self.c.snapshot(f"bwd L{l}: gathered+bucket")
            self._release_layer(l)
            shards_l = self.c.reduce_scatter([g.store[f"gbucket{l}"] for g in self.c.gpus])
            for gpu, s in zip(self.c.gpus, shards_l):
                gpu.drop(f"gbucket{l}")
                gpu.put(f"gs{l}", s.astype(LOW), "grads")
            self.c.snapshot(f"bwd L{l}: reduced+freed")

        def optim(gpu: VirtualGPU):  # update my shard in place; nothing to all-gather now
            for l in range(self.L):
                new = adam_update(gpu.store[f"master{l}"], gpu.store[f"m{l}"], gpu.store[f"v{l}"], gpu.store[f"gs{l}"], self.t, self.adam)
                gpu.store[f"p{l}"][...] = new
                gpu.drop(f"gs{l}")

        self.c.parallel(optim)
        self.c.snapshot("after optimizer")

    def full_params(self):
        # for verification only -- gather without logging communication
        return [np.concatenate([g.store[f"p{l}"] for g in self.c.gpus]) for l in range(self.L)]


TRAINERS = {
    "DP": DataParallelTrainer,
    "ZeRO-1": ZeRO1Trainer,
    "ZeRO-2": ZeRO2Trainer,
    "ZeRO-3": ZeRO3Trainer,
}
