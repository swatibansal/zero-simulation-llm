"""
Demo model: an MLP classifier with hand-written forward/backward.

Why hand-written backprop instead of an autograd framework?  Because ZeRO-3 needs to
*gather a layer's parameters right before they are used and drop them right after*,
and reduce-scatter a layer's gradient the moment it is produced.  With explicit
per-layer forward/backward functions the sharding logic is visible instead of being
hidden behind hooks.

Parameter storage follows the DeepSpeed/FSDP convention of one *flat* buffer per
layer, padded so it divides evenly into `world_size` equal chunks.  Rank r "owns"
chunk r of every layer.  In BF16 mixed precision the working copy is 16-bit
(NumPy has float16, not bfloat16, so float16 stands in for it); the master copy and
the Adam moments are float32, giving the classic 2 + 2 + 12 = 16 bytes / parameter.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

LOW = np.float16  # stand-in for bf16 working precision
HIGH = np.float32


@dataclass
class LayerLayout:
    index: int
    shapes: List[Tuple[int, ...]]  # [(in, out), (out,)]
    n_real: int  # real parameters
    n_padded: int  # padded to a multiple of world_size
    chunk: int  # n_padded // world_size

    def views(self, flat: np.ndarray) -> List[np.ndarray]:
        """Reshape the (padded) flat buffer of this layer into W, b views."""
        out, off = [], 0
        for s in self.shapes:
            n = int(np.prod(s))
            out.append(flat[off:off + n].reshape(s))
            off += n
        return out


class MLP:
    def __init__(self, sizes: List[int], world_size: int):
        """sizes = [d_in, h1, h2, ..., n_classes]"""
        self.sizes = sizes
        self.world_size = world_size
        self.layers: List[LayerLayout] = []
        for i, (a, b) in enumerate(zip(sizes[:-1], sizes[1:])):
            n = a * b + b
            padded = -(-n // world_size) * world_size
            self.layers.append(LayerLayout(i, [(a, b), (b,)], n, padded, padded // world_size))
        self.n_layers = len(self.layers)
        self.n_params = sum(l.n_real for l in self.layers)
        self.n_padded = sum(l.n_padded for l in self.layers)

    # ---- initialisation ---------------------------------------------------------------
    def init_layer(self, l: int, rng: np.random.Generator) -> np.ndarray:
        lay = self.layers[l]
        a, b = lay.shapes[0]
        flat = np.zeros(lay.n_padded, dtype=HIGH)
        W, bias = lay.views(flat)
        scale = np.sqrt(2.0 / a) if l < self.n_layers - 1 else 0.1 / np.sqrt(a)  # He init; small logits at start
        W[...] = rng.standard_normal((a, b), dtype=HIGH) * scale
        bias[...] = 0.0
        return flat

    # ---- per-layer compute (what a GPU kernel would do) ---------------------------------
    def layer_forward(self, l: int, flat_low: np.ndarray, x: np.ndarray):
        """Returns (output, cache).  Last layer is linear (logits), others ReLU."""
        W, b = self.layers[l].views(flat_low)
        z = x.astype(HIGH) @ W.astype(HIGH) + b.astype(HIGH)
        if l < self.n_layers - 1:
            out = np.maximum(z, 0)
            cache = (x, out > 0)
        else:
            out = z
            cache = (x, None)
        return out.astype(LOW), cache

    def layer_backward(self, l: int, flat_low: np.ndarray, cache, dout: np.ndarray):
        """Returns (dx, flat_grad) where flat_grad has the layer's padded flat layout."""
        x, mask = cache
        lay = self.layers[l]
        W, b = lay.views(flat_low)
        dz = dout.astype(HIGH)
        if mask is not None:
            dz = dz * mask
        g = np.zeros(lay.n_padded, dtype=HIGH)
        gW, gb = lay.views(g)
        gW[...] = x.astype(HIGH).T @ dz
        gb[...] = dz.sum(0)
        dx = dz @ W.astype(HIGH).T
        return dx.astype(LOW), g

    def layer_flops(self, l: int, batch: int, backward: bool = False) -> int:
        a, b = self.layers[l].shapes[0]
        fwd = 2 * batch * a * b
        return 2 * fwd if backward else fwd  # backward = dX and dW matmuls

    # ---- loss -----------------------------------------------------------------------------
    @staticmethod
    def softmax_xent(logits: np.ndarray, y: np.ndarray, scale: float = 1.0):
        """Mean cross-entropy and dlogits.  `scale` lets callers divide by world_size so that
        summing gradients across ranks yields the global-batch mean gradient."""
        z = logits.astype(HIGH)
        z = z - z.max(1, keepdims=True)
        p = np.exp(z)
        p /= p.sum(1, keepdims=True)
        n = len(y)
        loss = -np.log(p[np.arange(n), y] + 1e-12).mean()
        dl = p
        dl[np.arange(n), y] -= 1
        dl *= scale / n
        return float(loss), dl.astype(LOW)


def make_dataset(n: int, d_in: int, n_classes: int, seed: int = 0):
    """Synthetic classification task: labels come from a random 'teacher' network,
    so the loss has real structure to learn (it should go down)."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d_in), dtype=HIGH)
    W1 = rng.standard_normal((d_in, 64), dtype=HIGH)
    W2 = rng.standard_normal((64, n_classes), dtype=HIGH)
    y = (np.tanh(X @ W1) @ W2).argmax(1)
    return X.astype(LOW), y.astype(np.int64)
