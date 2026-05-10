# Paper Experiment Commands

Run these from the repo root.

## 1. Multi-seed stability

```bash
python3 scripts/run_paper_experiments.py multiseed --seeds 42,43,44,45,46
```

## 2. Ablation study

```bash
python3 scripts/run_paper_experiments.py ablations
```

## 3. Explanation faithfulness

```bash
python3 scripts/run_paper_experiments.py faithfulness --faithfulness-top-k 3
```

## 4. Threshold sweep

```bash
python3 scripts/run_paper_experiments.py threshold
```

## 5. Robustness to text perturbations

```bash
python3 scripts/run_paper_experiments.py robustness
```

## 6. Cross-dataset schema check

Requires the TwiBot-22 files to exist under `data/raw/TwiBot-22/`.

```bash
python3 scripts/run_paper_experiments.py cross-dataset --dataset-config config/twibot22.yaml
```

All outputs are saved under `reports/paper_experiments/`.

Each experiment now also writes paper-friendly artifacts:
- `summary.md`
- one or more `.png` comparison plots
- the existing CSV / JSON outputs

## Postprocess already-finished runs

Useful if an experiment finished before plotting code was added.

```bash
python3 scripts/run_paper_experiments.py postprocess --experiment multiseed --output-dir reports/paper_experiments/multiseed
python3 scripts/run_paper_experiments.py postprocess --experiment ablations --output-dir reports/paper_experiments/ablations
python3 scripts/run_paper_experiments.py postprocess --experiment faithfulness --output-dir reports/paper_experiments/faithfulness
python3 scripts/run_paper_experiments.py postprocess --experiment threshold --output-dir reports/paper_experiments/threshold
python3 scripts/run_paper_experiments.py postprocess --experiment robustness --output-dir reports/paper_experiments/robustness
python3 scripts/run_paper_experiments.py postprocess --experiment cross_dataset --output-dir reports/paper_experiments/cross_dataset
```
