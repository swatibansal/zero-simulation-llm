"""Train the demo model with DP / ZeRO-1 / ZeRO-2 / ZeRO-3 on 32 virtual GPUs and print the comparison."""
import argparse

import numpy as np

from zero_sim import TRAINERS, MLP, AdamConfig, Cluster, comm_volume_bytes, fmt_bytes, make_dataset, model_state_bytes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", type=int, default=32)
    ap.add_argument("--gpus-per-node", type=int, default=8)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--micro-batch", type=int, default=32)
    ap.add_argument("--sizes", type=int, nargs="+", default=[256, 1024, 1024, 1024, 1024, 10])
    a = ap.parse_args()

    model = MLP(a.sizes, a.gpus)
    X, y = make_dataset(a.gpus * a.micro_batch * a.steps, a.sizes[0], a.sizes[-1], seed=1)
    print(f"{a.gpus} virtual GPUs ({a.gpus // a.gpus_per_node} nodes x {a.gpus_per_node}), "
          f"Psi = {model.n_padded:,} params, global batch {a.gpus * a.micro_batch}, {a.steps} steps\n")
    hdr = f"{'stage':8s}{'loss':>8s}{'peak/GPU':>11s}{'theory':>10s}{'vs DP':>7s}{'comm/rank':>11s}{'theory':>10s}{'wall/step':>11s}"
    print(hdr); print("-" * len(hdr))
    ref, dp_peak = None, None
    for name, T in TRAINERS.items():
        c = Cluster(a.gpus, a.gpus_per_node)
        tr = T(c, model, AdamConfig(lr=2e-3)); tr.setup()
        for s in range(a.steps):
            i = s * a.gpus * a.micro_batch
            st = tr.step(X[i:i + a.gpus * a.micro_batch], y[i:i + a.gpus * a.micro_batch])
        if ref is None:
            ref, dp_peak = tr.full_params(), st.peak_bytes_per_gpu
        same = all(np.array_equal(p, q) for p, q in zip(ref, tr.full_params()))
        print(f"{name:8s}{st.loss:>8.4f}{fmt_bytes(st.peak_bytes_per_gpu):>11s}{fmt_bytes(model_state_bytes(model.n_padded, a.gpus, name).total):>10s}"
              f"{dp_peak / st.peak_bytes_per_gpu:>6.1f}x{fmt_bytes(st.comm_bytes_per_rank):>11s}"
              f"{fmt_bytes(comm_volume_bytes(model.n_padded, a.gpus, name)):>10s}{st.wall_seconds:>9.2f} s   weights == DP: {same}")


if __name__ == "__main__":
    main()
