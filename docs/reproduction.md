# Reproduction guide

**Status: nothing has been run on Edge-IIoTset by this repository.** Every benchmark number in the article is, here, a *published reference* (`references/reported/`), or *recomputed from printed aggregates* (`iiot-qelm audit`). A result is called *measured* only when `runs/<id>/` holds the score matrices (hash-checked) and run metadata that let anyone recompute it.

## Three outcome labels

| Label | Meaning | Where |
|---|---|---|
| published reference | transcribed from the article, attributed to table/row/column | `references/reported/*.json` |
| recomputed from printed aggregates | derived from other printed numbers (arithmetic only, no model run) | `iiot-qelm audit`, `docs/paper-audit.md` |
| measured in this repository | produced by a run whose scores/timings are saved and recomputable | `runs/<id>/` (+ `iiot-qelm report`) |
| fixture | a run on invented data; certifies the code path only | `runs/demo-fixture/` and any run flagged `synthetic_fixture` |

## Environment

The article reports Python 3.9, TensorFlow 2.11, scikit-learn 1.2, NumPy 1.24, pandas 1.5.3 on an RTX 3080 (10 GB), i7-10700, 32 GB RAM. This repository is a PyTorch implementation; the tested runtime was Python 3.10.11, NumPy 1.26.4, pandas 2.3.3, scikit-learn 1.7.2, torch 2.13 (CPU), xgboost 3.2.0 on Windows 11. These are **deviations** from the reported versions and are written into every `run.json`. For GPU, install a CUDA build of torch and set `train.device: cuda` in the config.

## Copy-paste commands (the ones marked * were executed for the release)

```bash
pip install -e ".[xgboost,dev]"                                   # *
iiot-qelm demo                                                    # * offline, CPU, invented fixture
iiot-qelm demo --gwo                                              # * adds the FAST-BUDGET GWO demo
iiot-qelm audit --out docs/paper-audit-generated.md               # * the paper audit, no data
pytest --cov=iiot_qelm                                            # *

# full-scale (needs the dataset; see docs/data.md)
iiot-qelm data check   --csv data/raw/DNN-EdgeIIoT-dataset.csv
iiot-qelm data prepare --csv data/raw/DNN-EdgeIIoT-dataset.csv --out data/prepared --task multiclass --feature-list my_48_features.json
iiot-qelm data prepare --csv data/raw/DNN-EdgeIIoT-dataset.csv --out data/prepared_bin --task binary --feature-list my_48_features.json
iiot-qelm train     --data data/prepared --config configs/multiclass.yaml --run-dir runs/mc-fixed          # published fixed hyperparameters, 5 seeds
iiot-qelm tune      --data data/prepared --config configs/gwo_full.yaml  --run-dir runs/mc-fixed          # full GWO (very expensive)
iiot-qelm tune      --data data/prepared --fast --run-dir runs/mc-fast                                    # * reduced budget, flagged
iiot-qelm train     --data data/prepared --config configs/multiclass.yaml --gwo-config runs/mc-fixed/gwo/multiclass.json --run-dir runs/mc-gwo
iiot-qelm baselines --data data/prepared --config configs/multiclass.yaml --run-dir runs/mc-fixed          # ten comparators
iiot-qelm ablate    --data data/prepared --config configs/multiclass.yaml --run-dir runs/mc-fixed          # seven ablations + full
iiot-qelm evaluate  --run-dir runs/mc-fixed --task multiclass                                              # * recompute from saved scores
iiot-qelm report    --run-dir runs/mc-fixed --task multiclass --figures                                    # * tables + figures
iiot-qelm profile   --run-dir runs/mc-fixed --data data/prepared --seed 42                                 # * latency, params, MACs
iiot-qelm predict   --bundle runs/mc-fixed/seed42/full/bundle --data data/prepared --csv new_flows.csv --out predictions.csv   # *
iiot-qelm estimate  --rows 1527736                                                                         # * runtime estimate
```

Runs are **resumable**: a finished (model, seed) pair is skipped on rerun. A full-scale command without its input stops with the exact missing item (`missing: prepared data under ... To fix: run iiot-qelm data prepare ...`).

## Runtime (transparent, rough)

`iiot-qelm estimate` measured **1.9 optimiser steps/s (batch 32) on this CPU** for the Phase-1 network; extrapolated linearly to 1,527,736 training rows that is about **6 hours per epoch**, i.e. up to ~640 hours for 100 epochs, before early stopping (patience 15). The micro-benchmark includes per-epoch evaluation passes and is a crude upper bound, but the conclusion is robust: **full-scale Phase 1 is not practical on a CPU; use a GPU.** The full GWO fits the encoder five times (one per fold) and then solves a QELM per candidate per fold (up to 1,020 candidates x 5 folds); QELM solves are cheap relative to encoder training but not free at 1.2 M rows x 1,024 units. This tool never launches paid compute; nothing here was run on rented hardware.

## What is needed for an exact rerun

1. The labelled flow CSV (provider terms: docs/data.md) and the 48-column list (unpublished).
2. A GPU for Phase 1 at full scale.
3. For an *exact* comparison with the article: the run metadata the authors did not release (per-run scores / confusion matrices, split ids, the file actually used). Without them the discrepancies listed in docs/paper-audit.md (A-01, A-03, A-06, A-08) cannot be resolved by anyone.

## Reading the numbers

* Five seeds on one dataset and one outer split describe **training variability**, not generalisation.
* A workstation latency is **not** an embedded-gateway latency.
* The same outer split is used by every model and seed; only initialisation/shuffling differ across seeds.
