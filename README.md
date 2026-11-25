# iiot-qelm: Attention-QELM-GWO companion implementation

Independent, installable Python companion to **"Attention-guided quantum-inspired extreme learning machine with grey wolf optimization for rare-attack detection in IIoT networks"**, Shahiduzzaman, Sourav, Kabtia, Shammah, Islam, Chakraborty, Rahman, Islam, Tonoy, Adnan, *Discover Artificial Intelligence* 6:1258 (2026), DOI [10.1007/s44163-026-01704-3](https://doi.org/10.1007/s44163-026-01704-3).

| | |
|---|---|
| **Authors' own repository** | https://github.com/drmohammedadnan110-ai/Attention-QELM-GWO-IIoT-IDS (examined at `fc486ac`; **not copied**: [docs/UPSTREAM.md](docs/UPSTREAM.md)) |
| **Dataset** | Edge-IIoTset (Ferrag et al.): [Kaggle](https://www.kaggle.com/datasets/mohamedamineferrag/edgeiiotset-cyber-security-dataset-of-iot-iiot/data), DOI [10.21227/mbc1-1h68](https://doi.org/10.21227/mbc1-1h68), CC BY-NC-SA 4.0. Not redistributed |
| **Maintainer** | Partha Chakraborty ([@parthachk64-cyber](https://github.com/parthachk64-cyber)) |

> **Honest status.** This repository has **not** been run on Edge-IIoTset. No benchmark result is reproduced here. Every number from the article is either a *published reference* (`references/reported/`) or *recomputed from printed aggregates* (`iiot-qelm audit`). Runs on the built-in invented fixture certify the code path only.

It lets you follow one saved flow through the encoder and classifier, compare the method with all ten baselines, check the minority-class results, and tell exactly which numbers were published and which were independently measured. It works on saved tabular flow records only: no traffic generation, probing or device access.

## Quick start (CPU, no data, no downloads)

```bash
pip install -e ".[xgboost,dev]"        # Python >= 3.9; torch is the only heavy dependency
iiot-qelm demo                          # invented fixture: prepare -> fit small encoder + QELM -> save -> reload -> predict -> report + figures
iiot-qelm demo --gwo                    # also runs the FAST-BUDGET GWO demo (reduced; clearly marked)
iiot-qelm audit                         # recompute the article's derived numbers; needs no data
pytest --cov=iiot_qelm                  # 136 tests
```

Without installing: `PYTHONPATH=src python -m iiot_qelm.cli demo`.

## What is implemented

| Area | Where |
|---|---|
| Data: schema/aliases, raw validation, Sec. 2.2 preprocessing fitted on training rows only, immutable stratified 80/20 split (seed 7), manifests | `src/iiot_qelm/data/` |
| Attention encoder (8 tokens x 32, 8 heads), numerically stable focal loss, seeded QELM with closed-form ridge head, two-phase pipeline, save/load bundle | `models/` |
| Grey Wolf Optimizer (7-D, alpha/beta/delta, 20 wolves, 50 iterations, patience 10) with fold-safe encoder caching | `optim/` |
| Ten baselines (LR, NB, DT, RF, XGBoost, CNN-1D, LSTM, GRU, BiLSTM, DenseNet-1D) and the seven Table-11 ablations | `baselines/`, `models/variants.py` |
| Metrics, ROC, confusion matrices, Friedman, paired tests + Holm, efficiency, plots, report builder | `evaluation/`, `experiments/report.py` |
| Data-independent audit of the article | `evaluation/audit.py` -> [docs/paper-audit.md](docs/paper-audit.md) |
| Table/figure coverage (source, module, command, status) | [docs/coverage.md](docs/coverage.md) |

Docs: [data](docs/data.md) · [architecture](docs/architecture.md) · [reproduction](docs/reproduction.md) · [paper audit](docs/paper-audit.md) · [upstream inspection](docs/UPSTREAM.md) · [GWO contract](docs/gwo-contract.md).

## Published vs measured

* `references/reported/*.json`: values **transcribed** from the article, attributed to table/row/column. Never read by training, inference or table builders.
* `runs/<run-id>/`: values **measured** by this code, each with hash-checked score matrices, `run.json` (kind, git SHA, environment, dataset manifest hash, split fingerprint, seeds, reduced-budget flag). Metrics are recomputed from the saved scores every time. See [docs/reproduction.md](docs/reproduction.md).
* `iiot-qelm audit`: arithmetic recomputed from printed aggregates. Findings include a 19,000-record gap in Table 1 (A-01), a mean of the printed per-class F1 (0.98152) that differs from the printed macro-F1 (0.9859) (A-08), and a Friedman statistic that cannot exceed 50.00 for 5 seeds x 11 models while the Introduction states 54.98 (A-11). Table 10's p-values are reproduced exactly by a paired t-test + Holm and its effect sizes by a pooled-SD Cohen's d (A-12), which shows arithmetic consistency only.

## Full-scale experiments

Need the dataset (your own download, provider terms) and, realistically, a GPU: `iiot-qelm estimate` measured ~1.9 optimiser steps/s on the CPU used for development. Commands, configs, resumability and runtime estimates are in [docs/reproduction.md](docs/reproduction.md). A command whose data is absent stops with the exact missing item.

## What was reconstructed

The article is silent on several details (token expansion, attention output width, QELM scale/bias, role of theta, Phase-1 head, validation fraction, ...). Each choice is listed with its alternative in [docs/architecture.md](docs/architecture.md). Notably the global angle theta is distributionally inert under the released convention, and both regularisers are ridge terms.

## About

Maintained by Partha Chakraborty. His role in the published work, as stated in the article: reviewing the implementation workflow, checking the revised code structure, and supporting the analysis of the newly added results during revision. Software authorship is claimed here only for code that he reviews, adopts and commits under his own identity.

Cite the **article** (preferred): see [CITATION.cff](CITATION.cff), which also carries separate metadata for this software.

## Licence

MIT for this code ([LICENSE](LICENSE)); see [NOTICE](NOTICE) for third-party and dataset terms.
