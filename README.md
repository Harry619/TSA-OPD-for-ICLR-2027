# TSA-OPD-for-ICLR-2027
Controlled study of routing rules for entropy-aware on-policy distillation: coverage transfer is budget-limited, not routing-limited. verl-based training, evaluation, and analysis code.

# Precision or Coverage? — Code Repository (Anonymous)

Anonymous repository accompanying the ICLR 2026 submission
**"Precision or Coverage? Characterizing the Trade-off in Entropy-Aware
On-Policy Distillation of Reasoning Models"**.

This repository contains the full training, evaluation, and analysis code for
our controlled design-space study of routing rules in entropy-aware on-policy
distillation (OPD), built on a strictly reproduced EOPD protocol.

## What this study does

In on-policy distillation, a routing rule decides which tokens receive
coverage transfer (forward KL) from the teacher. We fix the protocol of
Entropy-Aware On-Policy Distillation (EOPD) — Qwen3-1.7B-Base student,
Qwen3-8B teacher (non-thinking), MATH training set, six math benchmarks,
identical trajectory budgets — and vary only the routing mechanism:

| Variant | Routing rule | Launcher |
|---|---|---|
| `opd` | Clipped reverse-KL only (no coverage transfer) | `reproduction/run_train.sh opd` |
| `eopd` | Hard entropy threshold gate (baseline) | `reproduction/run_train.sh eopd` |
| `soft_entropy` | Soft continuous entropy gate | `reproduction/run_tsa_train.sh soft_entropy` |
| `entropy_js` | Entropy gate × teacher–student JS divergence | `reproduction/run_tsa_train.sh entropy_js` |
| `entropy_js_mass` | Above × teacher top-k mass, annealed coefficient | `reproduction/run_tsa_train.sh entropy_js_mass` |
| `utility` / `psu` | Trajectory-utility / posterior-smoothed utility gates | `reproduction/run_utility_train.sh` |
| `branch` | Branch-value routing with teacher service | `reproduction/run_branch_train.sh` |
| CTS | Cover-then-sharpen curriculum over the coverage budget | `reproduction/run_cts_curriculum.sh` |

The main finding is that **coverage transfer is budget-limited, not
routing-limited**: utility-conditioned gates remove forward-KL mass precisely
from the low-success problems whose marginal Pass@k value is highest, while a
soft continuous entropy gate is statistically equivalent to EOPD on macro
Avg@8 at zero extra inference cost.

## Repository layout

```
verl/                  Training framework (fork; see "Provenance" below)
  trainer/ppo/tsa_opd.py        TSA-OPD routing utilities (new module)
  trainer/ppo/core_algos.py     OPD/EOPD/TSA losses and routing gates (modified)
  trainer/ppo/ray_trainer.py    Teacher top-k / entropy plumbing (modified)
  workers/actor/dp_actor.py     Loss integration (modified)
  workers/config/actor.py       Routing configuration fields (modified)
  workers/fsdp_workers.py       Teacher forward for entropy/top-k (modified)
reproduction/          All experiment code for this study
  PROTOCOL.md            Frozen reproduction protocol (hyperparameters, seeds)
  DATA_MANIFEST.md       Data sources, row counts, and SHA-256 hashes
  UTILITY_TSA_OPD.md     Utility/branch routing design notes
  run_*.sh               Training / evaluation / queue launchers
  tests/                 Unit tests for the routing implementations
  requirements-lock.txt  Pinned training environment
  conda-explicit.txt     Conda explicit spec of the training environment
  analysis/              Aggregated result summaries (JSON/Markdown)
reinforce/             Final-wave launchers and environment freezes
  env-freezes/           Exact pip freezes (training / eval / data hosts)
vendor/                (Not shipped — see "Evaluation harness" below)
docs/                  Upstream framework documentation
examples/, recipe/, tests/, docker/, scripts/   Upstream framework tree
```

## Provenance

- The training framework tree is the official EOPD repository (a verl fork),
  pinned at commit `a09cc1fa3a457b61515ca1b5f3c3fff2252eccbb`, with our
  routing extensions applied on top. The initial protocol diff is preserved
  at `reproduction/pinned-source-diff.patch`.
- The evaluation harness is Qwen2.5-Math pinned at commit
  `a45202bd16f1ec06f433442dc1152d0074773465` (fetched separately, see below).
- verl upstream documentation is kept under `docs/` (including
  `docs/verl_upstream_README.md`).

## Setup

Tested on Linux with 2×48GB GPUs per run (also ran on 2×80GB); CUDA 12.4,
Python 3.10.

```bash
conda create -n eopd-paper python=3.10 -y
conda activate eopd-paper
# torch 2.7.1 / vllm 0.10.0 / transformers 4.53.2 / ray 2.43.0 /
# flash-attn 2.7.4.post1 — exact pins:
pip install -r reproduction/requirements-lock.txt
pip install -e .
```

A full conda explicit spec is at `reproduction/conda-explicit.txt`; pip
freezes of every host are under `reinforce/env-freezes/`.
`reproduction/build_flash_attention.sh` documents how the flash-attn wheel
was built (SHA-256 in `reproduction/flash_attn_wheel_sha256.txt`).

## Models and data (not shipped)

Model weights and datasets are **not** included in this repository.

- Student: `Qwen/Qwen3-1.7B-Base` → place under `models/Qwen3-1.7B-Base/`
  (`reproduction/setup_model_cache.sh`).
- Teacher: `Qwen/Qwen3-8B` (Hugging Face cache layout; point the `TEACHER`
  variable in the launchers at your snapshot).
- Data: build the exact training/evaluation parquet files with
  `reproduction/prepare_data.py` (and `reproduction/prepare_deepmath_hard.py`
  for the hard-math extension). Expected row counts and SHA-256 hashes are
  listed in `reproduction/DATA_MANIFEST.md`. Launchers expect
  `data/math/{train,test}.parquet` under the repository root.

## Quick start (smoke test)

```bash
# 1-step training smoke for the soft entropy gate on 2 GPUs
bash reproduction/run_tsa_train.sh soft_entropy smoke 0,1 42

# OPD / EOPD reproduction smokes
bash reproduction/run_train.sh opd  smoke
bash reproduction/run_train.sh eopd smoke
```

Formal (full) runs replace `smoke` with `formal`. The launcher interface is

```
run_utility_train.sh {eopd|opd|soft_entropy|utility|psu|branch|eopd_value} \
                     {math|hardmix} {smoke|pilot|formal} GPU_IDS [SEED]
```

Multi-seed and stage-gated queues are in
`reproduction/run_utility_multiseed_queue.sh`,
`reproduction/run_formal_pipeline.sh`, and `reinforce/run_formal_pipeline.sh`.
The cover-then-sharpen curriculum is launched by
`reproduction/run_cts_curriculum.sh` (+ `run_cts_stage2_warm.sh`).

## Evaluation

Evaluation uses the pinned Qwen2.5-Math pipeline in a separate environment
(`reproduction/requirements-eval-lock.txt`):

```bash
git clone https://github.com/QwenLM/Qwen2.5-Math vendor/qwen2.5-math-a45202bd
cd vendor/qwen2.5-math-a45202bd && git checkout a45202bd16f1ec06f433442dc1152d0074773465
bash reproduction/setup_vendor.sh   # applies the adapter patch and copies eval data

# Evaluate a merged checkpoint on the six benchmarks (8 samples/problem)
bash reproduction/run_eval.sh MODEL_PATH RUN_NAME 0,1
# Pass@k spectra (128 samples) on AMC23/AIME24/AIME25
bash reproduction/run_eval_passk128.sh MODEL_PATH RUN_NAME 0,1
```

Protocol: zero-shot `qwen25-math-cot` prompts, temperature 1.0, top-p 0.8,
max generation length 8192, seed 42; rule-based scoring.

## Analysis

- `reproduction/passk_summary.py` — Avg@k / Pass@k tables from eval outputs
- `reproduction/paired_bootstrap.py` — paired bootstrap confidence intervals
- `reproduction/analyze_psu.py`, `reproduction/analyze_opd_eopd_passk.py` —
  routing/utility diagnostics
- `reproduction/summarize_utility_multiseed.py` — multi-seed aggregation
- Aggregated summaries: `reproduction/analysis/`

## Tests

```bash
python -m pytest reproduction/tests/
```

Covers the TSA routing math (`test_tsa_opd.py`), utility gates
(`test_tsa_utility.py`), PSU (`test_tsa_psu.py`), value-weighted EOPD
(`test_eopd_value.py`), and hard-mix reward routing
(`test_deepmath_reward_routing.py`).

## License

Apache License 2.0 (inherited from the upstream framework); see `LICENSE`
and `Notice.txt`.



