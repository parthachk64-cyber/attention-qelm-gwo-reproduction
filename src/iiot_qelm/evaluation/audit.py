"""Data-independent audit of the published article. Recomputes derived quantities from OTHER printed numbers and reports, without
correcting anything, where the printed values agree or disagree. Needs no dataset and no run. Outcome label for every line:
``recomputed_from_printed_aggregates`` (never ``measured``)."""
from __future__ import annotations

import math
from typing import Callable, Dict, List

import numpy as np
from scipy import stats as sps

from .. import reported as R
from . import stats as ST

OUTCOME = "recomputed_from_printed_aggregates"
REGISTRY: List[Callable[[], dict]] = []


def check(fn):
    REGISTRY.append(fn)
    return fn


def _res(id_, status, title, detail, severity="info", **data):
    return {"id": id_, "status": status, "severity": severity, "title": title, "detail": detail, "outcome": OUTCOME, "data": data}


@check
def a01_table1_class_sum():
    r = R.rows("table1")
    tot = R.load("table1")["printed_total"]
    s_tot, s_tr, s_te = (sum(x[k] for x in r) for k in ("total_records", "train_80", "test_20"))
    d = tot["total_records"] - s_tot
    ok = d == 0
    return _res("A-01", "consistent" if ok else "inconsistent", "Table 1 class totals vs the printed total",
                f"15 class rows sum to {s_tot:,} (train {s_tr:,}, test {s_te:,}); printed totals are {tot['total_records']:,} / {tot['train']:,} / {tot['test']:,}. "
                f"Row sum is lower by {d:,} (train by {tot['train'] - s_tr:,}, test by {tot['test'] - s_te:,}). Origin unresolved: needs the file-level class histogram.",
                "major", class_sum=s_tot, printed_total=tot["total_records"], difference=d)


@check
def a02_table1_rows_train_test():
    bad, fr = [], []
    for x in R.rows("table1"):
        if x["train_80"] + x["test_20"] != x["total_records"]:
            bad.append((x["class"], x["train_80"] + x["test_20"] - x["total_records"]))
        fr.append(round(x["train_80"] / x["total_records"], 4))
    return _res("A-02", "consistent" if not bad else "inconsistent", "Table 1 per-row train + test = total",
                "every row adds up" if not bad else f"rows whose train+test differs from the total: {bad}", "minor", train_fraction_range=[min(fr), max(fr)])


@check
def a03_percent_shares():
    t1, t2 = {x["class"]: x for x in R.rows("table1")}, {x["class"]: x for x in R.rows("table2")}
    tot = R.load("table1")["printed_total"]["total_records"]
    s = sum(x["total_records"] for x in R.rows("table1"))
    mism = []
    for c, x in t1.items():
        calc = round(100 * x["total_records"] / tot, 2)
        if abs(calc - x["pct_of_dataset"]) > 0.005:
            mism.append((c, x["pct_of_dataset"], calc))
    udp1, udp2 = t1["DDoS_UDP"]["pct_of_dataset"], t2["DDoS_UDP"]["pct_of_total"]
    return _res("A-03", "inconsistent" if mism else "consistent", "DDoS_UDP share: 5.31% (Table 1) vs 6.31% (Table 2)",
                f"Table 1 prints {udp1}%, Table 2 prints {udp2}% for the same 120,532 records. Over the printed total {tot:,} the share is {100 * 120532 / tot:.2f}%; over the class-row sum {s:,} it is {100 * 120532 / s:.2f}%. "
                f"Table 1 percentages that disagree with count/printed-total: {mism}. Origin unresolved.", "major", mismatches=mism)


@check
def a04_attack_total_and_pct_attack():
    t2 = R.rows("table2")
    pt = R.load("table2")["printed_total"]["attack_total"]
    attack_sum = sum(x["records"] for x in t2 if x["class"] != "Normal")
    mism = [(x["class"], x["pct_of_attack_only"], round(100 * x["records"] / pt, 2)) for x in t2 if x["class"] != "Normal" and abs(round(100 * x["records"] / pt, 2) - x["pct_of_attack_only"]) > 0.006]
    return _res("A-04", "inconsistent" if attack_sum != pt else "consistent", "Attack-only total and shares (Table 2)",
                f"the 14 attack rows sum to {attack_sum:,}; the printed attack total is {pt:,} (difference {pt - attack_sum:,}). The 'percent of attack only' column agrees with the printed attack total to within 0.01 pp (listed: {mism or 'none'}). Normal + printed attack total = {1367959 + pt:,} = the printed dataset total, so the missing {pt - attack_sum:,} records are attack records that no class row carries.",
                "major", attack_sum=attack_sum, printed=pt)


@check
def a05_imbalance_ratio():
    r = 1367959 / 1734
    return _res("A-05", "consistent" if r > 788 else "inconsistent", "Normal:MITM imbalance ratio", f"{r:.1f}:1 (article: 'exceeds 788 : 1')")


@check
def a06_binary_internal_consistency():
    n_norm = next(x["test_20"] for x in R.rows("table1") if x["class"] == "Normal")
    tot_test = R.load("table1")["printed_total"]["test"]
    n_att = tot_test - n_norm
    out, bad = [], []
    for x in R.rows("table6"):
        p, rc, f1, fpr, acc = x["precision"], x["recall"], x["f1"], x["fpr"], x["test_acc"]
        f1c = 2 * p * rc / (p + rc)
        tp, fn, fp = rc * n_att, (1 - rc) * n_att, fpr * n_norm
        prec_impl = tp / (tp + fp)
        acc_impl = 1 - (fp + fn) / (n_att + n_norm)
        out.append({"model": x["model"], "f1_from_PR": round(f1c, 4), "f1_printed": f1, "precision_implied": round(prec_impl, 4), "precision_printed": p,
                    "accuracy_implied": round(acc_impl, 4), "accuracy_printed": acc})
        if abs(prec_impl - p) > 0.002:
            bad.append(x["model"])
    prop = next(o for o in out if o["model"] == "Proposed Model")
    return _res("A-06", "inconsistent" if bad else "consistent", "Table 6: precision / recall / FPR / accuracy vs the Table 1 test composition",
                f"With the test partition from Table 1 ({n_norm:,} Normal, {n_att:,} attack by difference from the printed test total), the printed recall and FPR imply a precision and an accuracy. "
                f"Proposed model: implied precision {prop['precision_implied']} vs printed {prop['precision_printed']}; implied accuracy {prop['accuracy_implied']} vs printed {prop['accuracy_printed']}. "
                f"Rows whose implied precision differs from the printed value by more than 0.002: {bad}. F1 from printed P and R matches the printed F1 to rounding in every row. "
                f"Cause unresolved (a different test composition than Table 1, or non-matching run aggregation); the per-run confusion matrix would settle it.", "major", rows=out)


@check
def a07_false_alarm_counts():
    n = 273592
    t6 = {x["model"]: x for x in R.rows("table6")}
    calc = {m: round(t6[m]["fpr"] * n) for m in ("Proposed Model", "DenseNet", "XGBoost")}
    text = {"Proposed Model": 1176, "DenseNet": 2408, "XGBoost": 3911}
    ok = all(abs(calc[m] - text[m]) <= 20 for m in calc)
    return _res("A-07", "consistent" if ok else "inconsistent", "False-alarm counts in the text vs FPR x 273,592", f"FPR x Normal-test-rows = {calc}; text says {text}.")


@check
def a08_table8_mean_vs_table7():
    t8 = R.rows("table8")
    mean = float(np.mean([x["proposed_f1"] for x in t8]))
    printed = next(x["macro_f1"] for x in R.rows("table7") if x["model"] == "Proposed Model")
    return _res("A-08", "inconsistent" if abs(mean - printed) > 5e-5 else "consistent", "Mean of the 15 per-class F1 (Table 8) vs macro-F1 (Table 7)",
                f"arithmetic mean of the 15 printed proposed per-class F1 values = {mean:.5f}; Table 7 prints macro-F1 {printed}. Difference {printed - mean:+.5f}. "
                f"Macro-F1 over classes IS the unweighted mean of per-class F1, so the two tables cannot come from the same run and aggregation unless the printed values were rounded or averaged differently. "
                f"Resolving it needs the per-class precision/recall (or the confusion matrix) of the exact run behind Table 7, and the run list behind Table 8.", "major", mean=mean, printed=printed)


@check
def a09_table8_deltas():
    mism = []
    for x in R.rows("table8"):
        best = max(x["xgboost_f1"], x["densenet_f1"], x["bilstm_f1"])
        calc = round(100 * (x["proposed_f1"] - best), 2)
        if abs(calc - x["diff_vs_best_baseline_pp"]) > 0.011:
            mism.append((x["class"], x["diff_vs_best_baseline_pp"], calc))
    ddos = next(x for x in R.rows("table8") if x["class"] == "DDoS_HTTP")
    return _res("A-09", "consistent" if not mism else "inconsistent", "Table 8 'difference vs best baseline' recomputed",
                f"disagreements: {mism or 'none'}. DDoS_HTTP: proposed {ddos['proposed_f1']} vs DenseNet {ddos['densenet_f1']} = {100 * (ddos['proposed_f1'] - ddos['densenet_f1']):+.2f} pp. "
                f"Only XGBoost, DenseNet and BiLSTM are printed; the other seven comparators are not in Table 8.")


@check
def a10_table9_aggregates():
    mism = []
    for x in R.rows("table9"):
        s = ST.mean_sd_ci(x["runs_1_to_5"])
        if abs(s["mean"] - x["mean_macro_f1"]) > 6e-5 or abs(s["sd"] - x["sd"]) > 6e-5 or abs(s["ci_low"] - x["ci95"][0]) > 1.5e-4 or abs(s["ci_high"] - x["ci95"][1]) > 1.5e-4:
            mism.append((x["model"], {"mean": round(s["mean"], 5), "sd": round(s["sd"], 5), "ci": [round(s["ci_low"], 4), round(s["ci_high"], 4)]}, x["mean_macro_f1"], x["sd"], x["ci95"]))
    return _res("A-10", "consistent" if not mism else "inconsistent", "Table 9 mean / sample SD / Student-t 95% CI (df=4) recomputed from the five printed runs",
                f"disagreements beyond rounding: {mism or 'none'}.", "minor")


@check
def a11_friedman():
    m = {x["model"]: x["runs_1_to_5"] for x in R.rows("table9")}
    f = ST.friedman(m)
    claims = {c[0]: c[2] for c in R.load("text_claims")["rows"]}
    return _res("A-11", "inconsistent", "Friedman statistic",
                f"from the printed runs: chi2 = {f['chi2']:.2f}, df = {f['df']}, p = {f['p']:.3g}; the maximum attainable with {f['n_blocks']} blocks and {f['n_models']} models is {f['max_attainable_chi2']:.2f} "
                f"(identical rank order in every seed), which is exactly the printed 50.00 / 2.67e-7. The Introduction (contribution 5) states chi2 = {claims['friedman_intro']} with p < 0.0001, which is not what Sec. 3.7 and the Conclusion state. "
                f"A tie-free chi2 above 50.00 is impossible for this design, so 54.98 cannot come from these five-by-eleven data.", "major", **{k: v for k, v in f.items() if k != "mean_rank"})


@check
def a12_table10():
    m = {x["model"]: x["runs_1_to_5"] for x in R.rows("table9")}
    t10 = {x["comparison_proposed_vs"]: x for x in R.rows("table10")}
    scores = {k: v for k, v in m.items()}
    res: Dict[str, list] = {}
    for test in ("paired_t", "welch_t"):
        res[test] = ST.planned_vs_reference(scores, "Proposed Model", test)
    lines = []
    for r_pair, r_welch in zip(res["paired_t"], res["welch_t"]):
        p = t10[r_pair["comparator"]]
        lines.append({"vs": r_pair["comparator"], "printed_diff": p["mean_diff_macro_f1"], "diff_from_runs": round(r_pair["mean_diff"], 4), "printed_p_holm": p["holm_corrected_p"],
                      "paired_t_p_holm": r_pair["p_holm"], "welch_p_holm": r_welch["p_holm"], "printed_d": p["cohens_d"], "d_pooled": round(r_pair["d_pooled"], 2), "d_z": round(r_pair["d_z"], 2)})
    match_pooled = sum(abs(l["d_pooled"] - l["printed_d"]) <= 0.006 for l in lines)
    match_p = sum(abs(l["paired_t_p_holm"] - l["printed_p_holm"]) <= 0.0051 * l["printed_p_holm"] + 1e-9 for l in lines)
    return _res("A-12", "reproducible_from_printed_runs", "Table 10 recomputed from the printed Table 9 runs",
                f"Mean differences agree with Table 10 to rounding. A two-sided PAIRED t-test on the five seed differences (df = 4) with Holm correction over the ten planned comparisons "
                f"reproduces the printed corrected p-value in {match_p}/10 rows; Cohen's d computed with the POOLED SD of the two models reproduces the printed d in {match_pooled}/10 rows "
                f"(the paired d_z, mean difference / SD of differences, is a different and larger number; Welch's t-test does not reproduce the p-values). So the test and effect-size formula behind Table 10 are identified, "
                f"but this shows only that the printed numbers are arithmetically consistent with the printed runs; it is not a new model run. With n = 5 seeds on one dataset and one split the tests describe training variability; "
                f"an exact Wilcoxon signed-rank test could not go below p = {2 / 2 ** 5:.4f} (two-sided).", "info", rows=lines)


@check
def a13_ablation_deltas():
    t11 = R.rows("table11")
    full = t11[0]["macro_f1"]
    mism = [(x["variant"], x["delta_macro_f1"], round(x["macro_f1"] - full, 4)) for x in t11[1:] if abs(round(x["macro_f1"] - full, 4) - x["delta_macro_f1"]) > 1e-4]
    t8 = {x["class"]: x["proposed_f1"] for x in R.rows("table8")}
    same = (t11[0]["mitm_f1"], t11[0]["ransomware_f1"], t11[0]["os_fingerprinting_f1"]) == (t8["MITM"], t8["Ransomware"], t8["OS_Fingerprinting"])
    return _res("A-13", "consistent" if not mism and same else "inconsistent", "Table 11 deltas and the full-model row", f"delta disagreements: {mism or 'none'}; full-model per-class F1 equals Table 8: {same}.")


@check
def a14_train_test_gap_claim():
    t6 = R.rows("table6")
    gaps = {x["model"]: round(x["train_acc"] - x["test_acc"], 4) for x in t6}
    dl = {m: g for m, g in gaps.items() if m in ("CNN-1D", "LSTM", "GRU", "BiLSTM", "DenseNet", "Proposed Model")}
    smallest = min(dl, key=dl.get)
    return _res("A-14", "inconsistent" if smallest != "Proposed Model" else "consistent", "'Smallest train-test accuracy gap among the deep models' (Sec. 3.1)",
                f"gaps (train - test): {dl}. Smallest: {smallest}. The article says the proposed gap of 0.0014 is the smallest.", "minor", gaps=dl)


@check
def a15_efficiency_arithmetic():
    t12 = R.rows("table12")
    tp = []
    for x in t12:
        if x["gpu_ms_per_sample"] and x["gpu_samples_per_s"]:
            tp.append((x["model"], round(1000 / x["gpu_ms_per_sample"]), x["gpu_samples_per_s"]))
    bad = [t for t in tp if abs(t[1] - t[2]) > 0.01 * t[2]]
    gwo_evals = (20 * 38 + 20, 20 * 31 + 20)
    return _res("A-15", "consistent" if not bad else "inconsistent", "Table 12 throughput vs latency, and the GWO budget",
                f"1000 / GPU ms per sample vs printed samples/s: disagreements {bad or 'none'}. 42 + 25 = 67 min (printed). With early stopping after 38 and 31 iterations at N=20 the search "
                f"performed at most {gwo_evals[0]} (multi-class) and {gwo_evals[1]} (binary) candidate evaluations, not the nominal 1,000 stated in Sec. 2.5.1 (candidates vs fold fits are not distinguished in the article).")


@check
def a16_gwo_space_claim():
    return _res("A-16", "consistent", "Grid-search size claim", f"ten values for each of seven dimensions = 10^7 = {10 ** 7:,} configurations (article: 10^7).")


@check
def a17_parameter_convention():
    return _res("A-17", "unresolved", "Parameter count convention (Table 12)",
                "The article reports 0.21 M 'trainable or solved' parameters. For the architecture in Sec. 2.4 (3x512 QELM) the frozen random hidden weights alone are ~0.66 M and the QELM solved head is 7.7 k for 15 classes; "
                "which of encoder / hidden / output weights the 0.21 M counts is not stated. `iiot-qelm profile` reports each category separately.", "minor")


def run_all() -> List[dict]:
    return [fn() for fn in REGISTRY]


def to_markdown(results: List[dict]) -> str:
    lines = ["# Paper audit (recomputed from printed aggregates; no data, no run)", "",
             "Every line recomputes a derived quantity from OTHER numbers the article prints. Nothing is corrected. Outcome label: `recomputed_from_printed_aggregates`.", "",
             "| ID | Severity | Status | Finding |", "|---|---|---|---|"]
    for r in results:
        lines.append(f"| {r['id']} | {r['severity']} | {r['status']} | **{r['title']}.** {r['detail']} |")
    return "\n".join(lines) + "\n"
