# Upstream project: what was inspected

The article's Data availability section names the authors' repository. It is the technical reference this repository was compared with. **No upstream code is copied here**; everything was written against the published method.

| Field | Value |
|---|---|
| Repository | https://github.com/drmohammedadnan110-ai/Attention-QELM-GWO-IIoT-IDS |
| Examined | 2026-11-25, default branch, whole tree read (33 tracked files) |
| Commit examined | `fc486ac5b7cadf17b0b961d68016beef5cc9376f` (2026-06-08, "feat: initial release of Attention-QELM-GWO IIoT IDS framework"), the only commit; no tags or releases |
| Licence | MIT, "Copyright (c) 2026" naming seven people: Md Shahiduzzaman, Md Sultanul Arefin Sourav, Maria Kabtia, Md Azharul Islam, MD Mizanur Rahman, Mazharul Islam, Mohammed Adnan. The published article has ten authors; the licence and the upstream `CITATION.cff` do not name Ramisa Samin Shammah, Partha Chakraborty or Rahat Bin Aziz Tonoy |
| Citation in its README | "Manuscript under review", seven authors. The published ten-author record is used in this repository (`CITATION.cff`) |
| Code reused here | **None.** No source, config, schema or asset is copied. If any is adopted later it must be listed in `NOTICE` with its licence text |

## What the authors released at this revision

Status words: **available**, **unavailable at this revision**, **unverified** (could not be established without data this repository does not have).

| Item | Status | Detail |
|---|---|---|
| Feature schema / authoritative 48-column list | **unavailable** | No column list anywhere. `preprocessing.py` derives the columns by rule and writes `metadata.json` at run time; nothing asserts that 48 survive |
| Preprocessing code | available | `src/data/preprocessing.py`. Differences from the article: metadata columns removed by **substring** match (`'mqtt.msg'` would also match `mqtt.msgtype` and `mqtt.msg_decoded_as` if those columns exist: unverified); constant and duplicate columns detected on the **whole** file before splitting; the split is stratified on the **task** label (binary vs 15-class give different outer splits) |
| Tensor shapes | available (docstrings, `docs/ARCHITECTURE.md`) | but the default `EncoderConfig()` **raises** `ValueError: bottleneck_dim must equal num_tokens * key_dim (got 64 != 8 * 32)`, so the shipped encoder and `tests/test_smoke.py::test_attention_encoder_output_shape` fail as released. Verified by executing the dataclass with a stubbed TensorFlow |
| QELM solver | available | `src/models/qelm.py`: `solve(H^T H + (l1 + l2) I, H^T T)`; global angle theta **added** to every sampled angle; weights scaled by sqrt(2 / fan_in); biases `0.1 * U(-1, 1)`; softmax over regression outputs. Neither the scaling nor the bias rule is in the article |
| Tuning (GWO) code | available | `src/models/gwo.py`, `proposed_model.run_gwo_search`. Fitness folds reuse embeddings from a Phase-1 encoder trained on the **entire** training partition, so fold-validation labels reach the encoder (leakage); leaders are updated independently and can coincide; evaluation count is N(T+1), not N T |
| Phase-1 early stopping | available | `fit_phase1` monitors `val_loss` only if validation data is passed; `src/train.py` passes none, so early stopping monitors **training** loss |
| Predictions, per-run scores, training histories, checkpoints | **unavailable** | none committed; `results/tables/README.md` says tables are "regenerated rather than checked in" |
| Split IDs / split file | **unavailable** | only a seed (7); no row identifiers |
| `docs/RESULTS.md` | available | reproduces the article's printed numbers; it is a transcription, **not** run output |
| Ablation configs | **unavailable** | `scripts/run_ablation.sh` calls `configs/ablation/*.yaml`, which do not exist, and passes flags `src.train` does not define |
| Aggregation script | **unavailable** | `run_experiments.sh` ends with `python -m scripts.aggregate_runs`; no such module |
| README commands | partly broken | `python -m src.evaluate --config ... --checkpoint runs/final/best.h5`: `evaluate.py` accepts only `--data-dir --run-dir --out` |
| Raw CSV identity | **unverified** | README uses the placeholder `data/raw/edge_iiot.csv`; the download script uses Kaggle slug `...-of-iotiiot` while the article's URL is `...-of-iot-iiot` (the public Kaggle metadata endpoint answers 200 for the article's slug and 403 for upstream's) |
| Baseline settings | available | `src/baselines/*.py` (Keras models; the numbers in the article used TensorFlow 2.11) |
| Dependency pins | available | `requirements.txt` pins TensorFlow 2.11.1, xgboost 1.7.6 (article lists scikit-learn 1.2, NumPy 1.24, pandas 1.5.3 and Python 3.9) |

## The dataset file

Public Kaggle metadata, read 2026-11-25 (dataset version 5, last updated 2022-03-18, licence CC BY-NC-SA 4.0, archive 11,248,814,873 bytes):

* `Edge-IIoTset dataset/Selected dataset for ML and DL/DNN-EdgeIIoT-dataset.csv`, 1,217,413,981 bytes
* `Edge-IIoTset dataset/Selected dataset for ML and DL/ML-EdgeIIoT-dataset.csv`, 82,184,390 bytes

**Which file the authors used is not stated.** By size only the 1.2 GB file can hold roughly 1.9 million rows of 60+ columns (82 MB would allow at most about 43 bytes per row); this is an inference, not a verification. A SHA-256 is not published by the provider; `iiot-qelm data check` records the checksum of whatever file you supply. The DOI https://doi.org/10.21227/mbc1-1h68 resolves (HTTP 200).

## Why an independent implementation

The released code cannot run as shipped (encoder config), its tuning leaks fold labels, and it ships no artifacts that would let anyone recompute a published number. An independent implementation written from the article exposed the places where the article is silent (`docs/architecture.md`) and gives every measured number a recomputable trail.
