# Data

No dataset is redistributed, and the repository needs none for the demo, the audit, the docs or the tests.

## Source

Ferrag, Friha, Hamouda, Maglaras, Janicke. *Edge-IIoTset*. Kaggle: https://www.kaggle.com/datasets/mohamedamineferrag/edgeiiotset-cyber-security-dataset-of-iot-iiot/data. DOI: https://doi.org/10.21227/mbc1-1h68.

* Licence per the Kaggle page: **CC BY-NC-SA 4.0** (non-commercial, share-alike). Read the provider terms before use.
* Storage: the whole archive is 11.2 GB (pcaps included). You only need one labelled flow CSV: the 1.2 GB `DNN-EdgeIIoT-dataset.csv` or the 82 MB `ML-EdgeIIoT-dataset.csv` (`Edge-IIoTset dataset/Selected dataset for ML and DL/`). Which one the authors used is not stated (docs/UPSTREAM.md).
* The repository works on **saved tabular flow records**. It needs no traffic generation, probing or device access.

## Acquire

```bash
# Option A: manual download (always works), then point --csv at the file.
# Option B: if you have Kaggle credentials of your own (never paste them into a chat or commit them):
iiot-qelm data download --out data/raw
```

## Validate and prepare

```bash
iiot-qelm data check --csv data/raw/DNN-EdgeIIoT-dataset.csv          # file name, bytes, SHA-256, rows, class histogram, missing classes
iiot-qelm data prepare --csv data/raw/DNN-EdgeIIoT-dataset.csv --out data/prepared --task multiclass --feature-list my_48_features.json
```

`my_48_features.json` (a JSON list of the 48 column names) is **yours to supply**: the article reports 48 retained features but publishes no list (docs/UPSTREAM.md). Without `--feature-list`, `prepare` refuses to continue unless the rule-derived set has exactly 48 columns; it will not drop arbitrary fields to reach 48. `--expected-features 0` accepts the rule-derived set and records its size in the manifest.

## Contract

| Step | Rule |
|---|---|
| Labels | canonical 15 names; aliases in `data/schema.py` (`DDos_HTTP`, `SQL_injection`, `Fingerprinting`, `Vulnerability_scanner`, `MTTM`...). Unknown labels are an error. Binary: Normal = 0, any attack = 1. `Attack_label` / `Attack_type` can never be features |
| Identifier/metadata columns | removed by **exact** name (`data/schema.py::METADATA_EXACT`), no data statistics used |
| Duplicates | hash-deduplicated on retained feature columns + label; identical-feature / different-label groups are audited and (default) dropped; row ids are content hashes; train and test ids are asserted disjoint |
| Outer split | stratified 80/20 on the **15-class** label, seed 7 (upstream README), shared by the binary and 15-class tasks and by every model and run seed; split fingerprint recorded |
| Fitted on training rows only | constant / duplicate / >40%-missing column drop, finite maxima (+inf -> max, -inf -> min), medians, modes, categorical codes (unseen -> reserved code), min-max scaling (test values clipped to [0, 1]), inverse-frequency class weights |
| Validation / early stopping | 10% drawn from the outer training partition only (the paper gives no fraction: implementer's choice) |
| Manifest | file name, bytes, SHA-256, row/column counts before/after each filter, class histograms, dropped columns, feature order, split fingerprint |

## What the article's Table 1 counts mean for your run

The printed class rows sum to 1,890,671, not the printed 1,909,671, and the shares conflict between Tables 1 and 2 (docs/paper-audit.md, A-01..A-04). The manifest of a real run lets you see which counts your file actually produces; this repository does not resolve the discrepancy without the file.
