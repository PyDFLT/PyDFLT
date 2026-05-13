# Reproducing *Sufficient Decision Proxies for Decision-Focused Learning* (IJCAI 2026)

This directory contains the driver script, configs, and portfolio data needed to reproduce the experiments in:

> Noah Schutte, Krzysztof Postek, Grigorii Veviurko, Neil Yorke-Smith.
> *Sufficient Decision Proxies for Decision-Focused Learning.* IJCAI 2026.

The reusable contributions (problems, decision makers, predictors, utilities) live in `src/pydflt/`. This directory only contains the experiment-specific runner and configs.

For the exact code snapshot used in the paper, see the top-level `README.md`.

## Contents

- `run.py` — entry point. Loads a problem config, applies a method override, runs over seeds.
- `configs/problems/{portfolio,wsmc,ptsp}.yml` — one config per problem with the shared training and evaluation setup.
- `configs/methods/{point,2_point,8_point,16_point,qp,pfl,residual_SAA}.yml` — one config per method; merged on top of the problem config at runtime.
- `data/portfolio_10_{seed}.npz` — pre-generated portfolio datasets for seeds 5–14. The `wsmc` and `ptsp` datasets are generated on the fly by the in-repo data loaders.

## Prerequisites

Install dependencies from the repo root using `uv`:

```bash
uv sync --all-extras --all-groups
```

(See the top-level `README.md` for full setup instructions.)

The experiments log to [Weights & Biases](https://wandb.ai/) by default (`use_wandb: true` in each problem config, with project names `portfolio_ijcai2026`, `wsmc_ijcai2026`, `ptsp_ijcai2026`). To run without W&B, set `use_wandb: false` in the relevant problem config.

## Running the experiments

All commands are run from the repository root. Each command runs one problem across the listed methods; the default seed range is `5 6 7 8 9 10 11 12 13 14` (override with `--seeds`).

**Portfolio:**

```bash
python experiments/sufficient-decision-proxies-ijcai2026/run.py \
    --problem portfolio \
    --methods pfl residual_SAA qp point 2_point 8_point 16_point
```

**Weighted set multi-cover (WSMC):**

```bash
python experiments/sufficient-decision-proxies-ijcai2026/run.py \
    --problem wsmc \
    --methods pfl residual_SAA qp point 2_point
```

**Probabilistic TSP (PTSP):**

```bash
python experiments/sufficient-decision-proxies-ijcai2026/run.py \
    --problem ptsp \
    --methods pfl residual_SAA qp point 2_point 8_point
```

To run a subset of seeds, pass `--seeds`:

```bash
python experiments/sufficient-decision-proxies-ijcai2026/run.py \
    --problem portfolio --methods qp --seeds 5 6 7
```

## How a run is configured

For each `(method, seed)` pair, `run.py`:

1. Loads the problem config (`configs/problems/<problem>.yml`).
2. Loads the method config (`configs/methods/<method>.yml`) and merges it onto the problem config.
3. Applies any problem-specific `method_overrides` block from the problem config (e.g. WSMC reduces `num_samples` to 2 for the `qp` method; portfolio disables prediction standardisation for the `*_point` methods because their bank-return logic depends on un-standardised values).
4. Sets the seed on every config key listed in `_keys_with_randomization` (defaults to `runner`, `problem`, `decision_maker`, `data`, `model`).
5. For the portfolio problem only, runs a pre-run hook that:
   - points `data.path` at `data/portfolio_10_<seed>.npz`,
   - computes the bank return rate as the median of the (`NUM_RELEVANT_SECURITY = 7`)-th best return across the training split, and writes it to `model.bank_return`,
   - and, for the `2_point` method, also sets the predictor `shift` to that bank return.
6. Calls `pydflt.utils.experiments.run(config)`.

## Where results land

Per-experiment outputs are written under `results/` (the path comes from `runner.experiments_folder` in each problem config). When `use_wandb` is enabled, runs additionally stream to the W&B project named in the problem config. Validation and test metrics tracked: `objective`, `abs_regret`, `rel_regret`, `sym_rel_regret` (the primary metric is `abs_regret`).

Note that results might be slightly different from the paper. This is due to: 1. Making the reproducibility more robust by adjusting the use of seeds in this version of the code. 2. A solver like Gurobi uses randomness that is uncontrollable, which means that each device will result in slightly different results (when there are multiple optimal decisions not always the same one is returned). Despite this results should not be significantly different, due to running 10 seeds in the experiments.

## Citing this code

If you use this code, please cite the paper and the software. See the top-level `README.md` "How to cite" section for the canonical BibTeX entries. The exact commit corresponding to the IJCAI 2026 paper is recorded there as well.
