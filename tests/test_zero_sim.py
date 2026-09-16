"""
The tests encode the claims the README makes.  If they pass, the simulator agrees with
the ZeRO paper on memory and communication, and sharding does not change the math.
"""
import numpy as np
import pytest

from zero_sim import (
    MLP, TRAINERS, AdamConfig, Cluster, comm_volume_bytes, make_dataset, model_state_bytes,
)

N = 32
SIZES = [64, 256, 256, 10]


def _run(stage, steps=2, seed=0):
    model = MLP(SIZES, N)
    X, y = make_dataset(N * 8 * steps, SIZES[0], SIZES[-1], seed=seed)
    c = Cluster(N)
    tr = TRAINERS[stage](c, model, AdamConfig(lr=2e-3))
    tr.setup()
    last = None
    for s in range(steps):
        i = s * N * 8
        last = tr.step(X[i:i + N * 8], y[i:i + N * 8])
    return model, c, tr, last


# ---- collectives ---------------------------------------------------------------------

def test_reduce_scatter_matches_sum():
    c = Cluster(N)
    arrs = [np.random.default_rng(r).standard_normal(N * 7).astype(np.float32) for r in range(N)]
    shards = c.reduce_scatter(arrs)
    full = np.sum(arrs, axis=0)
    for r in range(N):
        np.testing.assert_allclose(shards[r], full[r * 7:(r + 1) * 7], rtol=1e-4, atol=1e-5)
    rec = c.comm.records[-1]
    assert rec.elements == 7 * (N - 1)  # (N-1)/N of the data per rank


def test_all_gather_concatenates_in_rank_order():
    c = Cluster(N)
    shards = [np.full(3, r, dtype=np.float32) for r in range(N)]
    outs = c.all_gather(shards)
    expected = np.repeat(np.arange(N, dtype=np.float32), 3)
    for o in outs:
        np.testing.assert_array_equal(o, expected)


def test_all_reduce_is_reduce_scatter_plus_all_gather():
    c = Cluster(N)
    arrs = [np.random.default_rng(r).standard_normal(N * 4).astype(np.float32) for r in range(N)]
    outs = c.all_reduce(arrs)
    for o in outs:
        np.testing.assert_allclose(o, np.sum(arrs, axis=0), rtol=1e-4, atol=1e-5)
    kinds = [r.collective for r in c.comm.records]
    assert kinds == ["reduce_scatter", "all_gather"]


# ---- the ZeRO claims -----------------------------------------------------------------

@pytest.mark.parametrize("stage", ["ZeRO-1", "ZeRO-2", "ZeRO-3"])
def test_sharding_does_not_change_the_math(stage):
    _, _, dp, _ = _run("DP")
    _, _, zr, _ = _run(stage)
    for a, b in zip(dp.full_params(), zr.full_params()):
        np.testing.assert_array_equal(a, b)  # bitwise identical weights after 2 steps


@pytest.mark.parametrize("stage", ["DP", "ZeRO-1", "ZeRO-2", "ZeRO-3"])
def test_persistent_memory_matches_paper_formula(stage):
    model, c, tr, st = _run(stage, steps=1)
    est = model_state_bytes(model.n_padded, N, stage)
    resident = c.gpus[0].mem.live  # after the step: only persistent state remains
    assert resident["params"] == est.params
    assert resident["grads"] == 0  # gradients are consumed by the optimizer
    assert resident["master"] + resident["optim"] == est.optim
    assert resident["activations"] == 0 and resident["temp"] == 0  # no leaks


@pytest.mark.parametrize("stage", ["DP", "ZeRO-1", "ZeRO-2", "ZeRO-3"])
def test_comm_volume_matches_paper(stage):
    model, c, tr, st = _run(stage, steps=1)
    assert st.comm_bytes_per_rank == pytest.approx(comm_volume_bytes(model.n_padded, N, stage))


def test_memory_ladder_is_monotone():
    peaks = {s: _run(s, steps=1)[3].peak_bytes_per_gpu for s in TRAINERS}
    assert peaks["DP"] > peaks["ZeRO-1"] > peaks["ZeRO-2"] > peaks["ZeRO-3"]


def test_zero3_comm_is_1_5x_dp():
    dp = _run("DP", steps=1)[3].comm_bytes_per_rank
    z3 = _run("ZeRO-3", steps=1)[3].comm_bytes_per_rank
    assert z3 == pytest.approx(1.5 * dp)
