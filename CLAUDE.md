# CLAUDE.md — zero-sim

Context for Claude Code when working in this repository.

## What this repo is

A pure-NumPy simulator of distributed training on **32 virtual GPUs** (CPU threads, arranged as 4 nodes × 8).
It trains one small MLP four ways — plain data parallel (DP), ZeRO-1, ZeRO-2, ZeRO-3 — and counts every byte of
"HBM" and every byte on the "wire" so the memory/communication trade-offs of ZeRO can be *measured*, not just read
about. It is a learning project built from the ERA V5 session guide on distributed training; the README is the
write-up and is written in the author's first-person voice.

Invariants the code is built around (the tests enforce them — do not "fix" them away):

1. **All four trainers produce bitwise-identical weights after every step.** Sharding changes where bytes live,
   never the math. `tests/test_zero_sim.py::test_sharding_does_not_change_the_math`.
2. **Persistent per-GPU memory matches the ZeRO paper**: DP = 16Ψ, ZeRO-1 = (4 + 12/N)Ψ, ZeRO-2 = (2 + 14/N)Ψ,
   ZeRO-3 = 16Ψ/N. `test_persistent_memory_matches_paper_formula`.
3. **Communication per rank per step**: DP = ZeRO-1 = ZeRO-2 = 2Ψ, ZeRO-3 = 3Ψ (each ring collective moves
   (N−1)/N of its tensor). `test_comm_volume_matches_paper`, `test_zero3_comm_is_1_5x_dp`.
4. The memory ledger must never leak: after a step only `params`, `master`, `optim` are resident.

## Layout

```
zero_sim/cluster.py    VirtualGPU (memory ledger + storage + thread), Cluster, ring collectives, α-β cost model
zero_sim/model.py      MLP with hand-written forward/backward; flat per-layer buffers padded to N chunks
zero_sim/trainers.py   DataParallelTrainer → ZeRO1Trainer → ZeRO2Trainer → ZeRO3Trainer (inheritance chain)
zero_sim/analytic.py   closed-form ZeRO memory / comm formulas, used to cross-check the simulator
zero_sim_demo.ipynb    executed notebook (outputs committed on purpose — it is the deliverable)
run_demo.py            CLI summary table
tests/                 16 tests, ~2 s
figures/               PNGs exported from the notebook, referenced by README.md
```

## Commands

```bash
pip install -r requirements.txt
python -m pytest tests -q                 # must stay green; ~2 s
python run_demo.py --steps 2              # quick end-to-end sanity table
jupyter nbconvert --to notebook --execute --inplace zero_sim_demo.ipynb   # ~1 min; re-run after any code change
```

The notebook is generated from a builder script that is **not** in the repo (it lived in the Cowork session).
Edit `zero_sim_demo.ipynb` directly, or ask before regenerating it from scratch.

## Conventions and gotchas

- **BLAS must stay single-threaded.** `zero_sim/__init__.py` pins `OMP/OPENBLAS/MKL_NUM_THREADS=1` and calls
  `threadpoolctl`. 32 Python threads × multithreaded BLAS made steps ~50× slower. Import `zero_sim` before NumPy
  in new scripts, or keep the `threadpoolctl` fallback.
- `float16` stands in for `bfloat16` (NumPy has no bf16). Reductions inside collectives are done in fp32.
- Every allocation on a virtual GPU goes through `gpu.put(key, array, category)` / `gpu.drop(key)`. Never stash
  arrays in `gpu.store` without a ledger entry unless they are genuinely size-neutral scratch (prefixed `_`).
- Categories are fixed: `params, grads, master, optim, activations, temp`. Adding one requires updating
  `CATEGORIES`, the notebook's `cats` list, and `CAT_COLORS`.
- Collectives operate on lists of per-rank arrays and must be called from the host between `cluster.parallel()`
  phases — they are not thread-safe and are not meant to be.
- Loss is scaled by 1/N per rank so that the *sum* reduction equals the global-batch mean gradient.
- Keep README numbers in sync with `run_demo.py` output if the model size, N, or Adam config changes.
- Python ≥ 3.10; NumPy, matplotlib, threadpoolctl only. Do not add PyTorch — the point is that there is none.

## Git workflow

- Branch `main` is the submission. Work on feature branches and squash-merge; keep `main` linear.
- Commit messages: imperative, one line ≤ 72 chars, optional body explaining *why*. Prefix with the area when
  useful: `cluster:`, `trainers:`, `notebook:`, `readme:`, `tests:`.
- Before every commit: `python -m pytest tests -q`. If `zero_sim/` changed, also re-execute the notebook and
  re-export any figure whose plot changed (`figures/*.png` are referenced from the README).
- Never commit `__pycache__`, `.pytest_cache`, `.ipynb_checkpoints` (already in `.gitignore`).
- Do not rewrite history on `main` after it has been pushed.
- If the remote does not exist yet: `git remote add origin <url> && git push -u origin main`.

## Review checklist (when asked to review changes)

1. Do the four invariants above still hold? Run the tests; do not reason about it.
2. Does any change make a stage's memory or comm accounting *look* better than the paper formula says it should?
   That is almost certainly a ledger bug, not an optimisation.
3. Is the README's prose still true after the change (numbers, claims, figure references)?
4. Does the notebook still execute top to bottom with zero error outputs?
5. Are explanations written for the intended reader — a strong engineer who is new to ML systems — without
   hand-waving or unearned jargon?

## Things not to do

- Do not replace the hand-written backprop with autograd; the explicit per-layer calls are what make the ZeRO-3
  gather/release choreography visible.
- Do not change the ring collectives to `np.sum` shortcuts; the step-by-step ring is how the byte counts are earned.
- Do not edit the author's first-person voice in the README into third person or marketing copy.
