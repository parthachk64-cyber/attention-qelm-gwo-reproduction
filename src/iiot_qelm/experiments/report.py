"""Builds the Table 6-12 analogues from a run directory. Every number is recomputed from saved score matrices (hash-checked), never read
from ``metrics.json`` or from references/reported. Labels: ``measured`` (benchmark run) or ``fixture`` (code-path only). Reduced-budget runs
carry their flag. A fixture run is watermarked on every table."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from ..baselines import ALL_BASELINES, LABELS
from ..data import schema as S
from ..evaluation import metrics as M
from ..evaluation import stats as ST
from ..models.variants import ABLATIONS
from .runner import FIXTURE, recompute_metrics

PROPOSED = "full"


def _fmt(x, nd=4):
    return "n/a" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{nd}f}"


def _label(m: str) -> str:
    return "Proposed" if m == PROPOSED else (ABLATIONS[m]["label"] if m in ABLATIONS else LABELS.get(m, m))


def collect(run_dir, task: str) -> Dict[str, Dict[int, dict]]:
    out: Dict[str, Dict[int, dict]] = {}
    for sd in sorted(Path(run_dir).glob("seed*")):
        seed = int(sd.name[4:])
        for md in sorted(p for p in sd.iterdir() if (p / "scores_test.npz").is_file()):
            out.setdefault(md.name, {})[seed] = recompute_metrics(md, task)
    return out


def _table(header: List[str], rows: List[List[str]]) -> str:
    return "\n".join(["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"] + ["| " + " | ".join(r) + " |" for r in rows])


def build_report(run_dir, task: str, out_path: Optional[str] = None) -> str:
    rd = Path(run_dir)
    run = json.loads((rd / "run.json").read_text(encoding="utf-8"))
    res = collect(rd, task)
    names = S.class_names_for(task)
    lines = [f"# Report for run `{run['run_id']}` ({task})", ""]
    if run["kind"] == FIXTURE:
        lines += [f"> **{run.get('banner')}**", ""]
    lines += [f"Outcome label for every number below: **{'fixture (code path only)' if run['kind'] == FIXTURE else 'measured in this repository'}**; recomputed from saved scores "
              f"(hash-checked). git SHA `{run.get('git_sha')}`, split fingerprint `{(run['dataset'].get('split_fingerprint') or '')[:16]}`, seeds {run['seeds']}."]
    if run.get("reduced_budget"):
        lines += ["", "> **Reduced-budget run** (fewer GWO wolves/iterations, fewer folds, subsampling or fewer epochs than the full configuration)."]
    lines.append("")
    models = [m for m in [PROPOSED] + list(ALL_BASELINES) + [a for a in ABLATIONS if a != PROPOSED] if m in res]
    missing = [b for b in ALL_BASELINES if b not in res]
    if missing:
        lines += [f"Baselines not run here: {', '.join(LABELS[b] for b in missing)}.", ""]

    def mean(m, f):
        v = [f(r) for r in res[m].values()]
        v = [x for x in v if x is not None]
        return float(np.mean(v)) if v else None

    # ---- Table 6 / 7
    if task == "binary":
        lines += ["## Binary (Table 6 analogue; seed-mean)", "", _table(["model", "train acc*", "test acc", "precision", "recall", "F1", "FPR", "AUC-ROC"],
                  [[_label(m), _fmt(mean(m, lambda r: r.get("train_accuracy"))), _fmt(mean(m, lambda r: r["accuracy"])), _fmt(mean(m, lambda r: r["precision"])), _fmt(mean(m, lambda r: r["recall"])),
                    _fmt(mean(m, lambda r: r["f1"])), _fmt(mean(m, lambda r: r["fpr"])), _fmt(mean(m, lambda r: r["auc_roc"]))] for m in models]),
                  "", "*train accuracy is logged at fit time and is not recomputable from the saved test scores. FPR = FP/(FP+TN); AUC from continuous scores.", ""]
    else:
        lines += ["## 15-class (Table 7 analogue; seed-mean)", "", _table(["model", "train acc*", "test acc", "macro-P", "macro-R", "macro-F1", "weighted-F1", "macro OvR FPR", "OvR AUC"],
                  [[_label(m), _fmt(mean(m, lambda r: r.get("train_accuracy"))), _fmt(mean(m, lambda r: r["accuracy"])), _fmt(mean(m, lambda r: r["macro_precision"])),
                    _fmt(mean(m, lambda r: r["macro_recall"])), _fmt(mean(m, lambda r: r["macro_f1"])), _fmt(mean(m, lambda r: r["weighted_f1"])), _fmt(mean(m, lambda r: r["fpr_macro_ovr"])),
                    _fmt(mean(m, lambda r: r["auc_ovr_macro"]))] for m in models]), "",
                  "Macro means average over classes present in the test partition; weighted-F1 is reported separately and never substituted. FPR convention: unweighted mean of one-vs-rest FPR.", ""]
        # ---- Table 8
        shown = [m for m in (PROPOSED, "xgboost", "densenet1d", "bilstm") if m in res]
        pcs = {m: {c: mean(m, lambda r, c=c: r["per_class"][c]["f1"]) for c in names} for m in res}
        d_all = M.top_comparator_delta({m: pcs[m] for m in res if m == PROPOSED or m in ALL_BASELINES}, PROPOSED) if PROPOSED in res and len(res) > 1 else {}
        d_show = M.top_comparator_delta({m: pcs[m] for m in shown}, PROPOSED) if PROPOSED in shown and len(shown) > 1 else {}
        sup = {c: int(np.mean([r["per_class"][c]["support"] for r in res[models[0]].values()])) for c in names}
        rows = [[c, str(sup[c])] + [_fmt(pcs[m][c]) for m in shown] +
                [f"{d_show[c]['delta_pp']:+.2f} ({_label(d_show[c]['best_comparator'])})" if d_show.get(c, {}).get("delta_pp") is not None else "n/a",
                 f"{d_all[c]['delta_pp']:+.2f} ({_label(d_all[c]['best_comparator'])})" if d_all.get(c, {}).get("delta_pp") is not None else "n/a"] for c in names]
        lines += ["## Per-class F1 (Table 8 analogue; seed-mean)", "", _table(["class", "test support"] + [_label(m) for m in shown] + ["Δ pp vs strongest of the shown comparators", "Δ pp vs strongest of all run baselines"], rows), ""]
    # ---- Table 9 / 10
    seeds = sorted(set.intersection(*[set(res[m]) for m in models])) if models else []
    if len(seeds) >= 2:
        mf = {m: [res[m][s]["macro_f1"] for s in seeds] for m in models if m in ALL_BASELINES or m == PROPOSED}
        rows = []
        for m, v in mf.items():
            st = ST.mean_sd_ci(v)
            rows.append([_label(m)] + [_fmt(x) for x in v] + [_fmt(st["mean"]), _fmt(st["sd"], 5), f"[{_fmt(st['ci_low'])}, {_fmt(st['ci_high'])}]"])
        lines += [f"## Macro-F1 by seed (Table 9 analogue; seeds {seeds})", "", _table(["model"] + [f"seed {s}" for s in seeds] + ["mean", "SD (n-1)", "95% t-CI"], rows), "",
                  "Five seeds on one split and one dataset describe training variability only.", ""]
        if len(mf) >= 3:
            f = ST.friedman(mf)
            lines += [f"Friedman (models as treatments, seeds as blocks): chi2 = {f['chi2']:.2f}, df = {f['df']}, p = {f['p']:.3g} (maximum attainable chi2 for this design = {f['max_attainable_chi2']:.2f}).", ""]
        if PROPOSED in mf and len(mf) >= 2:
            pr = ST.planned_vs_reference(mf, PROPOSED, "paired_t")
            lines += ["Planned paired t-tests against the proposed model (two-sided, df = n-1), Holm-corrected over this family:", "",
                      _table(["comparison", "mean diff", "raw p", "Holm p", "Cohen d (pooled SD)", "d_z (paired)"], [[f"Proposed vs {_label(r['comparator'])}", _fmt(r["mean_diff"]), f"{r['p_raw']:.3g}", f"{r['p_holm']:.3g}",
                                                                                                                         f"{r['d_pooled']:.2f}", f"{r['d_z']:.2f}"] for r in pr]), ""]
    # ---- Table 11
    ab = [m for m in ABLATIONS if m in res]
    if len(ab) > 1 and PROPOSED in res:
        full = mean(PROPOSED, lambda r: r["macro_f1"])
        extra = [c for c in ("MITM", "Ransomware", "OS_Fingerprinting") if c in names]
        rows = [[_label(m), _fmt(mean(m, lambda r: r["accuracy"])), _fmt(mean(m, lambda r: r["macro_f1"]))] + [_fmt(mean(m, lambda r, c=c: r["per_class"][c]["f1"])) for c in extra] +
                ["-" if m == PROPOSED else f"{mean(m, lambda r: r['macro_f1']) - full:+.4f}"] for m in ab]
        lines += ["## Ablations (Table 11 analogue; seed-mean)", "", _table(["variant", "accuracy", "macro-F1"] + [f"{c} F1" for c in extra] + ["Δ macro-F1"], rows), ""]
    # ---- Table 12 + GWO
    prof = rd / "profile.json"
    if prof.is_file():
        p = json.loads(prof.read_text(encoding="utf-8"))
        lines += ["## Efficiency (Table 12 analogue; this hardware only)", "", "```json", json.dumps(p, indent=1)[:4000], "```", ""]
    g = rd / "gwo" / f"{task}.json"
    if g.is_file():
        j = json.loads(g.read_text(encoding="utf-8"))
        lines += ["## GWO search", "", f"best config {j['best_config']}, best CV macro-F1 {j['best_fitness']:.4f}; actual stopping iteration {j['iterations_run']} ({j['stop_reason']}); "
                  f"candidate evaluations {j['n_candidate_evaluations']} (fold fits {j['n_fold_fits']}); settings {j['settings']}; reduced budget: {j['reduced_budget']}.", ""]
    text = "\n".join(lines) + "\n"
    if out_path:
        Path(out_path).write_text(text, encoding="utf-8")
    return text
