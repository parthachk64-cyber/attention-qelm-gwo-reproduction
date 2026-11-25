# Coverage: every published table and figure

Availability reflects THIS repository's state: nothing has been run on Edge-IIoTset here.

| Item | Shows | Source of truth | Module | Command | Availability |
|---|---|---|---|---|---|
| Table 1 | class counts and 80/20 split | references/reported/table1.json; auditable per-class counts in data/manifest.json after `data prepare` | data/prepare.py, evaluation/audit.py (A-01..A-04) | `iiot-qelm audit`; `iiot-qelm data prepare` | audit: runs offline; measured counts: blocked (needs the dataset) |
| Table 2 | imbalance profile | same as Table 1 | data/prepare.py, evaluation/audit.py | `iiot-qelm audit` | audit: offline; measured: blocked (dataset) |
| Table 3 | GWO search space | optim/gwo.py BOUNDS (rendered in docs/gwo-contract.md) | optim/gwo.py | `iiot-qelm docs` | implemented |
| Table 4 | experimental configuration | configs/*.yaml; references/reported/table5.json | experiments/config.py | `iiot-qelm train --config ...` | implemented |
| Table 5 | implementation settings | configs/published_fixed.yaml | models/variants.py (PUBLISHED_FIXED) | `iiot-qelm train --config configs/published_fixed.yaml` | implemented; some details reconstructed (docs/architecture.md) |
| Table 6 | binary results | runs/<id>/seed*/<model>/scores_test.npz | evaluation/metrics.py, experiments/report.py | `iiot-qelm report --task binary` | blocked (dataset); fixture-tested |
| Table 7 | 15-class results | same | same | `iiot-qelm report --task multiclass` | blocked (dataset); fixture-tested |
| Table 8 | per-class F1 vs strongest baselines | same | evaluation/metrics.py (top_comparator_delta) | `iiot-qelm report --task multiclass` | blocked (dataset); fixture-tested |
| Table 9 | five-seed macro-F1, CI, Friedman rank | recomputed from printed runs (audit A-10, A-11); measured: runs/ | evaluation/stats.py | `iiot-qelm audit`; `iiot-qelm report` | audit offline; measured blocked (dataset, 11 models x 5 seeds) |
| Table 10 | planned paired tests, Holm, effect sizes | recomputed from printed runs (audit A-12) | evaluation/stats.py | `iiot-qelm audit`; `iiot-qelm report` | audit offline; measured blocked |
| Table 11 | ablations | runs/<id>/seed*/<variant>/ | models/variants.py | `iiot-qelm ablate` | blocked (dataset); every variant has an executable config test |
| Table 12 | parameters, ops, latency, memory | runs/<id>/profile.json | evaluation/efficiency.py | `iiot-qelm profile` | implemented; the paper's RTX 3080 numbers cannot be measured on other hardware |
| Figure 1 | IIoT overview | docs/architecture.md (original diagram of how saved flows enter the classifier) | evaluation/plots.py | `iiot-qelm docs` | implemented (original drawing) |
| Figure 2 | model architecture | docs/figures/architecture.png generated from the shape contract | evaluation/plots.py | `iiot-qelm docs` | implemented (original drawing) |
| Figure 3 | GWO algorithm | docs/gwo-contract.md | optim/gwo.py | `iiot-qelm docs` | implemented (pseudocode from the implemented contract) |
| Figure 4 | per-class ROC | scores_test.npz | evaluation/plots.py | `iiot-qelm report --figures` | blocked (dataset); fixture-tested |
| Figures 5-6 | accuracy / loss curves | history.json (actual epoch logs) | evaluation/plots.py | `iiot-qelm report --figures` | blocked (dataset); fixture-tested. Curves are train/validation; test is logged-only if requested |
| Figure 7 | confusion matrix | scores_test.npz (full-test counts + normalised) | evaluation/plots.py | `iiot-qelm report --figures` | blocked (dataset); fixture-tested. The article's figure is a subset; it is not reconstructed |
