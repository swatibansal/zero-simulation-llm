"""
Virtual GPU cluster.

A *virtual GPU* is a Python object that owns:
  * a memory ledger  (bytes currently resident, per category, plus high-water marks)
  * a storage dict   (the actual NumPy arrays that "live in HBM" of this rank)
  * a CPU thread     (compute for rank r really runs on its own thread)

The cluster wires N of them together (default 32 = 4 nodes x 8 GPUs) and provides
the collectives that real frameworks get from NCCL:

    broadcast, reduce_scatter, all_gather, all_reduce (= reduce_scatter + all_gather)

The collectives are implemented as *ring* algorithms step by step, so the bytes each
rank puts on the wire are counted exactly, and a simple alpha-beta cost model turns
those bytes into a modelled wall-clock time using different bandwidth for
intra-node (NVLink-like) and inter-node (InfiniBand-like) hops.

Nothing here is a GPU.  The point is that the *bookkeeping* -- who owns which
bytes, who sends what to whom, and when temporaries appear and disappear -- is
exactly the bookkeeping DeepSpeed ZeRO / PyTorch FSDP do on real hardware.
"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np

# --------------------------------------------------------------------------------------
# Memory ledger
# --------------------------------------------------------------------------------------

CATEGORIES = ("params", "grads", "master", "optim", "activations", "temp")


class MemoryLedger:
    """Tracks resident bytes per category on one virtual GPU and the peak total."""

    def __init__(self):
        self.live: Dict[str, int] = {c: 0 for c in CATEGORIES}
        self.peak_total = 0
        self.peak_by_cat: Dict[str, int] = {c: 0 for c in CATEGORIES}
        self.timeline: List[tuple] = []  # (label, {cat: bytes})

    def alloc(self, cat: str, nbytes: int):
        assert cat in CATEGORIES, cat
        self.live[cat] += int(nbytes)
        self.peak_by_cat[cat] = max(self.peak_by_cat[cat], self.live[cat])
        self.peak_total = max(self.peak_total, self.total())

    def free(self, cat: str, nbytes: int):
        self.live[cat] -= int(nbytes)
        assert self.live[cat] >= -1, f"negative memory in {cat}: {self.live[cat]}"

    def total(self) -> int:
        return sum(self.live.values())

    def snapshot(self, label: str):
        self.timeline.append((label, dict(self.live)))

    def reset_peaks(self):
        self.peak_total = self.total()
        self.peak_by_cat = dict(self.live)
        self.timeline = []


# --------------------------------------------------------------------------------------
# Virtual GPU
# --------------------------------------------------------------------------------------


class VirtualGPU:
    def __init__(self, rank: int, node: int, local_rank: int):
        self.rank = rank
        self.node = node
        self.local_rank = local_rank
        self.mem = MemoryLedger()
        self.store: Dict[str, np.ndarray] = {}
        self._sizes: Dict[str, tuple] = {}  # key -> (category, nbytes) recorded at put()
        self.compute_seconds = 0.0  # real wall-clock spent in this rank's compute thread
        self.flops = 0  # modelled floating point operations executed

    # convenience: allocate an array *and* account for it -------------------------------
    def put(self, key: str, arr: np.ndarray, cat: str):
        """'cudaMalloc' an array into this rank's HBM under a memory category."""
        if key in self.store:
            raise KeyError(f"rank {self.rank}: {key} already allocated")
        self.store[key] = arr
        self._sizes[key] = (cat, arr.nbytes)
        self.mem.alloc(cat, arr.nbytes)

    def drop(self, key: str):
        """'cudaFree' -- releases exactly the bytes that were charged at put() time."""
        self.store.pop(key)
        cat, nbytes = self._sizes.pop(key)
        self.mem.free(cat, nbytes)

    def __repr__(self):
        return f"VirtualGPU(rank={self.rank}, node={self.node}, local={self.local_rank})"


# --------------------------------------------------------------------------------------
# Communication cost model
# --------------------------------------------------------------------------------------


@dataclass
class HardwareProfile:
    """Rough H100-SXM-class numbers.  Only the *ratios* matter for the lessons here."""

    name: str = "H100-like, 8 GPU/node, 400Gb/s IB"
    peak_flops: float = 989e12  # dense BF16 tensor-core peak
    mfu: float = 0.40  # realistic model-FLOPs-utilisation
    intra_node_bw: float = 450e9  # bytes/s per direction, NVLink/NVSwitch-ish
    inter_node_bw: float = 50e9  # bytes/s, one 400Gb/s NIC per GPU
    intra_latency: float = 5e-6  # seconds per hop
    inter_latency: float = 20e-6
    hbm_bytes: int = 80 * 1024**3


@dataclass
class CommRecord:
    step: int
    phase: str
    collective: str
    elements: int  # elements moved per rank (total over the ring steps)
    bytes_per_rank: int
    spans_nodes: bool
    sim_seconds: float


class CommLog:
    def __init__(self):
        self.records: List[CommRecord] = []
        self.step = 0
        self.phase = ""

    def total_bytes_per_rank(self, step: Optional[int] = None) -> int:
        return sum(r.bytes_per_rank for r in self.records if step is None or r.step == step)

    def total_sim_seconds(self, step: Optional[int] = None) -> float:
        return sum(r.sim_seconds for r in self.records if step is None or r.step == step)

    def by_collective(self, step: Optional[int] = None) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for r in self.records:
            if step is None or r.step == step:
                out[r.collective] = out.get(r.collective, 0) + r.bytes_per_rank
        return out


# --------------------------------------------------------------------------------------
# Cluster + collectives
# --------------------------------------------------------------------------------------


class Cluster:
    def __init__(
        self,
        world_size: int = 32,
        gpus_per_node: int = 8,
        hw: HardwareProfile | None = None,
        comm_dtype_bytes: int = 2,  # gradients/params travel the wire as BF16
        threads: Optional[int] = None,
    ):
        assert world_size % gpus_per_node == 0
        self.N = world_size
        self.gpus_per_node = gpus_per_node
        self.hw = hw or HardwareProfile()
        self.comm_dtype_bytes = comm_dtype_bytes
        self.gpus: List[VirtualGPU] = [
            VirtualGPU(r, r // gpus_per_node, r % gpus_per_node) for r in range(world_size)
        ]
        self.pool = ThreadPoolExecutor(max_workers=threads or world_size)
        self.comm = CommLog()

    # ---- run something on every rank concurrently ----------------------------------
    def parallel(self, fn: Callable[[VirtualGPU], object], *args) -> list:
        def timed(gpu):
            t0 = time.perf_counter()
            out = fn(gpu, *args)
            gpu.compute_seconds += time.perf_counter() - t0
            return out

        return list(self.pool.map(timed, self.gpus))

    def snapshot(self, label: str):
        for g in self.gpus:
            g.mem.snapshot(label)

    # ---- cost model ------------------------------------------------------------------
    def _ring_time(self, ranks: List[int], bytes_per_step: int, steps: int) -> tuple[float, bool]:
        nodes = {self.gpus[r].node for r in ranks}
        spans = len(nodes) > 1
        # A ring that crosses node boundaries runs at the speed of its slowest link.
        bw = self.hw.inter_node_bw if spans else self.hw.intra_node_bw
        lat = self.hw.inter_latency if spans else self.hw.intra_latency
        return steps * (lat + bytes_per_step / bw), spans

    def _log(self, collective: str, elements_per_rank: int, ranks: List[int], steps: int, chunk_elems: int):
        bytes_per_rank = elements_per_rank * self.comm_dtype_bytes
        t, spans = self._ring_time(ranks, chunk_elems * self.comm_dtype_bytes, steps)
        self.comm.records.append(
            CommRecord(self.comm.step, self.comm.phase, collective, elements_per_rank, bytes_per_rank, spans, t)
        )

    # ---- collectives -----------------------------------------------------------------
    def broadcast(self, arrays: List[np.ndarray], root: int = 0) -> List[np.ndarray]:
        """Root's array is copied to every rank. Modelled as a ring/pipeline broadcast."""
        n = len(arrays)
        src = arrays[root]
        out = [src.copy() if r != root else src for r in range(n)]
        # ring broadcast: each non-root rank receives the whole tensor once, forwards once
        self._log("broadcast", src.size, list(range(n)), steps=n - 1, chunk_elems=src.size // (n - 1) + 1)
        return out

    def reduce_scatter(self, arrays: List[np.ndarray], op: str = "sum") -> List[np.ndarray]:
        """
        Ring reduce-scatter.  Input: one equal-length 1-D array per rank.
        Output: rank r receives chunk r of the *reduced* array (1/N of the data).

        Ring algorithm: N-1 steps; at each step every rank sends one chunk to its right
        neighbour and accumulates the chunk it receives from its left neighbour.
        Bytes per rank on the wire = (N-1)/N * len(array).
        """
        N = len(arrays)
        size = arrays[0].size
        assert size % N == 0, "array must be padded to a multiple of world size"
        chunk = size // N
        # reduction is done in fp32 exactly like NCCL accumulates in the higher precision
        acc = [[a[i * chunk:(i + 1) * chunk].astype(np.float32) for i in range(N)] for a in arrays]
        for s in range(N - 1):
            # rank r sends chunk (r-1-s) to rank r+1, which adds it into its own copy
            sends = [(r, (r - 1 - s) % N) for r in range(N)]
            payload = [acc[r][idx].copy() for r, idx in sends]
            for (r, idx), p in zip(sends, payload):
                acc[(r + 1) % N][idx] += p
        out = [acc[r][r] for r in range(N)]  # after N-1 steps rank r holds fully reduced chunk r
        if op == "mean":
            out = [o / N for o in out]
        self._log("reduce_scatter", chunk * (N - 1), list(range(N)), steps=N - 1, chunk_elems=chunk)
        return out

    def all_gather(self, shards: List[np.ndarray]) -> List[np.ndarray]:
        """
        Ring all-gather.  Input: rank r contributes shard r (all equal length).
        Output: every rank ends with the concatenation of all shards.
        Bytes per rank on the wire = (N-1)/N * full_size.
        """
        N = len(shards)
        chunk = shards[0].size
        have = [{r: shards[r]} for r in range(N)]
        for s in range(N - 1):
            # rank r forwards the chunk it received last step (initially its own)
            sends = [(r, (r - s) % N) for r in range(N)]
            payload = [have[r][idx] for r, idx in sends]
            for (r, idx), p in zip(sends, payload):
                have[(r + 1) % N][idx] = p
        out = [np.concatenate([have[r][i] for i in range(N)]) for r in range(N)]
        self._log("all_gather", chunk * (N - 1), list(range(N)), steps=N - 1, chunk_elems=chunk)
        return out

    def all_reduce(self, arrays: List[np.ndarray], op: str = "sum") -> List[np.ndarray]:
        """all-reduce == reduce-scatter followed by all-gather (the key identity)."""
        shards = self.reduce_scatter(arrays, op=op)
        return self.all_gather(shards)

    # ---- helpers ---------------------------------------------------------------------
    def reset(self):
        for g in self.gpus:
            g.mem.reset_peaks()
            g.compute_seconds = 0.0
            g.flops = 0
        self.comm = CommLog()

    def peak_bytes_per_gpu(self) -> int:
        return max(g.mem.peak_total for g in self.gpus)

    def modelled_compute_seconds(self) -> float:
        """Modelled compute time of the slowest rank (a synchronous step waits for it)."""
        return max(g.flops for g in self.gpus) / (self.hw.peak_flops * self.hw.mfu)

    def wall_compute_seconds(self) -> float:
        return max(g.compute_seconds for g in self.gpus)
