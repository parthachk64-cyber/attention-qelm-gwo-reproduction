"""Generates documentation that must not drift from the code: the architecture figure, the GWO search contract, and the
coverage table assigning every published table and figure to a source, module, command and availability status."""
from __future__ import annotations

from pathlib import Path

from .evaluation import plots
from .optim.gwo import BOUNDS, DIM_NAMES

# (item, what the article shows, source of truth here, module, command, availability)
COVERAGE = [
    ("Table 1", "class counts and 80/20 split", "references/reported/table1.json; auditable per-class counts in data/manifest.json after `data prepare`", "data/prepare.py, evaluation/audit.py (A-01..A-04)", "`iiot-qelm audit`; `iiot-qelm data prepare`", "audit: runs offline; measured counts: blocked (needs the dataset)"),
    ("Table 2", "imbalance profile", "same as Table 1", "data/prepare.py, evaluation/audit.py", "`iiot-qelm audit`", "audit: offline; measured: blocked (dataset)"),
    ("Table 3", "GWO search space", "optim/gwo.py BOUNDS (rendered in docs/gwo-contract.md)", "optim/gwo.py", "`iiot-qelm docs`", "implemented"),
    ("Table 4", "experimental configuration", "configs/*.yaml; references/reported/table5.json", "experiments/config.py", "`iiot-qelm train --config ...`", "implemented"),
    ("Table 5", "implementation settings", "configs/published_fixed.yaml", "models/variants.py (PUBLISHED_FIXED)", "`iiot-qelm train --config configs/published_fixed.yaml`", "implemented; some details reconstructed (docs/architecture.md)"),
    ("Table 6", "binary results", "runs/<id>/seed*/<model>/scores_test.npz", "evaluation/metrics.py, experiments/report.py", "`iiot-qelm report --task binary`", "blocked (dataset); fixture-tested"),
    ("Table 7", "15-class results", "same", "same", "`iiot-qelm report --task multiclass`", "blocked (dataset); fixture-tested"),
    ("Table 8", "per-class F1 vs strongest baselines", "same", "evaluation/metrics.py (top_comparator_delta)", "`iiot-qelm report --task multiclass`", "blocked (dataset); fixture-tested"),
    ("Table 9", "five-seed macro-F1, CI, Friedman rank", "recomputed from printed runs (audit A-10, A-11); measured: runs/", "evaluation/stats.py", "`iiot-qelm audit`; `iiot-qelm report`", "audit offline; measured blocked (dataset, 11 models x 5 seeds)"),
    ("Table 10", "planned paired tests, Holm, effect sizes", "recomputed from printed runs (audit A-12)", "evaluation/stats.py", "`iiot-qelm audit`; `iiot-qelm report`", "audit offline; measured blocked"),
    ("Table 11", "ablations", "runs/<id>/seed*/<variant>/", "models/variants.py", "`iiot-qelm ablate`", "blocked (dataset); every variant has an executable config test"),
    ("Table 12", "parameters, ops, latency, memory", "runs/<id>/profile.json", "evaluation/efficiency.py", "`iiot-qelm profile`", "implemented; the paper's RTX 3080 numbers cannot be measured on other hardware"),
    ("Figure 1", "IIoT overview", "docs/architecture.md (original diagram of how saved flows enter the classifier)", "evaluation/plots.py", "`iiot-qelm docs`", "implemented (original drawing)"),
    ("Figure 2", "model architecture", "docs/figures/architecture.png generated from the shape contract", "evaluation/plots.py", "`iiot-qelm docs`", "implemented (original drawing)"),
    ("Figure 3", "GWO algorithm", "docs/gwo-contract.md", "optim/gwo.py", "`iiot-qelm docs`", "implemented (pseudocode from the implemented contract)"),
    ("Figure 4", "per-class ROC", "scores_test.npz", "evaluation/plots.py", "`iiot-qelm report --figures`", "blocked (dataset); fixture-tested"),
    ("Figures 5-6", "accuracy / loss curves", "history.json (actual epoch logs)", "evaluation/plots.py", "`iiot-qelm report --figures`", "blocked (dataset); fixture-tested. Curves are train/validation; test is logged-only if requested"),
    ("Figure 7", "confusion matrix", "scores_test.npz (full-test counts + normalised)", "evaluation/plots.py", "`iiot-qelm report --figures`", "blocked (dataset); fixture-tested. The article's figure is a subset; it is not reconstructed"),
]


def write_gwo_contract(path) -> Path:
    rows = ["| dim | name | bounds | decoding |", "|---|---|---|---|"]
    dec = ["round half up, clip 1..5", "round half up, clip 64..1024", "floor; x = 4 -> index 3 of (sigmoid, tanh, relu, gelu)", "10 ** x", "10 ** x", "as is", "as is"]
    for i, n in enumerate(DIM_NAMES):
        rows.append(f"| {i} | {n} | [{BOUNDS[i, 0]:.6g}, {BOUNDS[i, 1]:.6g}] | {dec[i]} |")
    text = ["# GWO search contract (generated from `optim/gwo.py`)", "", *rows, "",
            "```", "initialise N wolves uniformly in the bounds; evaluate (iteration 0)", "leaders = three best DISTINCT positions seen so far (stable ties)",
            "for t = 1..T:  a = 2 (1 - t/T)", "    for each wolf X: for each leader L in (alpha, beta, delta):", "        A = 2 a r1 - a;  C = 2 r2;  X_L = L - A |C L - X|", "        X <- clip(mean(X_alpha, X_beta, X_delta), bounds)",
            "    evaluate decoded candidates (memoised by decoded config); update leaders", "    stop when `patience` consecutive iterations improve the best by <= tol", "return decoded alpha position", "```", "",
            "Candidate evaluations: N for the initial population + N per iteration (<= N (T+1) = 1,020 for N=20, T=50); each costs `n_folds` fold fits (the article's nominal 1,000 = N T does not distinguish the two).",
            "Fitness is five-fold macro-F1 with the encoder fitted per fold on that fold's fit partition only. GWO never runs at inference.", ""]
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(text), encoding="utf-8")
    return p


def write_coverage(path) -> Path:
    lines = ["# Coverage: every published table and figure", "", "Availability reflects THIS repository's state: nothing has been run on Edge-IIoTset here.", "",
             "| Item | Shows | Source of truth | Module | Command | Availability |", "|---|---|---|---|---|---|"]
    lines += ["| " + " | ".join(r) + " |" for r in COVERAGE]
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p


def build_docs(docs_dir="docs") -> dict:
    d = Path(docs_dir)
    return {"architecture_png": str(plots.plot_architecture(d / "figures" / "architecture.png")), "gwo_contract": str(write_gwo_contract(d / "gwo-contract.md")),
            "coverage": str(write_coverage(d / "coverage.md"))}
