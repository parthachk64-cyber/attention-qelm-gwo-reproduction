"""Regenerates references/reported/*.json: TRANSCRIPTIONS of the published article's tables (Discover AI 6:1258, version of record
s44163-026-01704-3.pdf). Values are typed exactly as printed. Printed oddities are preserved and marked in ``notes``; nothing is
corrected. These files are references for comparison and auditing; no training, inference or table builder reads them as results.

    python tools/build_reported.py
"""
import json
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "references" / "reported"
SRC = {"doi": "10.1007/s44163-026-01704-3", "article": "Discover Artificial Intelligence 6:1258 (2026)", "licence": "CC BY-NC-ND 4.0 (transcribed numbers only)"}


def dump(name, table, page, columns, rows, notes=None, **extra):
    d = {"source": dict(SRC, table=table, page=page), "label": "published_reference", "columns": columns, "rows": rows, "notes": notes or []}
    d.update(extra)
    (OUT / f"{name}.json").write_text(json.dumps(d, indent=1, ensure_ascii=False), encoding="utf-8")


# ---------------------------------------------------------------- Table 1
t1 = [("Normal", "Benign", "Mixed", 1367959, 71.65, 1094367, 273592, 0), ("DDoS_UDP", "Volumetric Flood", "UDP", 120532, 5.31, 96426, 24106, 1),
      ("DDoS_ICMP", "Volumetric Flood", "ICMP", 67391, 3.53, 53913, 13476, 1), ("SQL_Injection", "Injection Attack", "HTTP", 50421, 2.64, 40337, 10084, 1),
      ("DDoS_TCP", "Volumetric Flood", "TCP", 49651, 2.60, 39721, 9930, 1), ("DDos_HTTP", "Volumetric Flood", "HTTP", 48693, 2.55, 38954, 9739, None),
      ("Password", "Credential Attack", "MQTT", 38217, 2.00, 30574, 7643, 1), ("Vulnerability_Scanner", "Reconnaissance", "TCP", 31482, 1.65, 25186, 6296, 1),
      ("Port_Scanning", "Reconnaissance", "TCP", 28904, 1.51, 23123, 5781, 1), ("XSS", "Web Attack", "HTTP", 22734, 1.19, 18187, 4547, 1),
      ("Uploading", "Malware", "HTTP", 19856, 1.04, 15885, 3971, 1), ("Backdoor", "Persistent Access", "TCP", 17423, 0.91, 13938, 3485, 1),
      ("OS_Fingerprinting", "Reconnaissance", "ICMP", 14381, 0.75, 11505, 2876, 1), ("Ransomware", "Extortion Malware", "TCP", 11293, 0.59, 9034, 2259, 1),
      ("MITM", "Interception", "ARP", 1734, 0.09, 1387, 347, 1)]
dump("table1", "Table 1", 8, ["class", "attack_category", "protocol_layer", "total_records", "pct_of_dataset", "train_80", "test_20", "label"],
     [list(r) for r in t1], ["printed total row: 15 classes, 1,909,671 total, 1,527,736 train, 381,935 test",
                             "the 'Label' cell of the DDos_HTTP row is blank in the printed table",
                             "the Port_Scanning total is printed with a stray leading backtick: `28,904", "class name is printed DDos_HTTP here and DDoS_HTTP elsewhere"],
     printed_total={"total_records": 1909671, "train": 1527736, "test": 381935, "classes": 15})

# ---------------------------------------------------------------- Table 2
t2 = [("Normal", "Dominant", 1367959, 71.65, None, "Severely Dominant"), ("DDoS_UDP", "Very High", 120532, 6.31, 22.26, "High"),
      ("DDoS_ICMP", "High", 67391, 3.53, 12.44, "Moderate-High"), ("SQL_Injection", "Moderate", 50421, 2.64, 9.31, "Moderate"),
      ("DDoS_TCP", "Moderate", 49651, 2.60, 9.17, "Moderate"), ("DDoS_HTTP", "Moderate", 48693, 2.55, 8.99, "Moderate"), ("Password", "Moderate", 38217, 2.00, 7.06, "Moderate"),
      ("Vulnerability_Scanner", "Low", 31482, 1.65, 5.81, "Minor"), ("Port_Scanning", "Low", 28904, 1.51, 5.34, "Minor"), ("XSS", "Low", 22734, 1.19, 4.20, "Minor"),
      ("Uploading", "Low", 19856, 1.04, 3.67, "Minor"), ("Backdoor", "Very Low", 17423, 0.91, 3.22, "Rare"), ("OS_Fingerprinting", "Very Low", 14381, 0.75, 2.66, "Rare"),
      ("Ransomware", "Very Low", 11293, 0.59, 2.09, "Rare"), ("MITM", "Extremely Low", 1734, 0.09, 0.32, "Extremely Rare")]
dump("table2", "Table 2", 8, ["class", "frequency_tier", "records", "pct_of_total", "pct_of_attack_only", "imbalance_level"], [list(r) for r in t2],
     ["DDoS_UDP is printed 5.31% in Table 1 and 6.31% in Table 2", "printed attack total: 541,712 records, 28.35% of the dataset"], printed_total={"attack_total": 541712, "attack_pct": 28.35})

# ---------------------------------------------------------------- Table 3 / 4 / 5 (settings)
dump("table3", "Table 3", 14, ["hyperparameter", "symbol", "domain", "type"],
     [["Hidden layers", "L", "[1, 5]", "Integer"], ["Neurons per layer", "n_h", "[64, 1024]", "Integer"], ["Activation function", "f", "{sigmoid, tanh, ReLU, GELU}", "Categorical"],
      ["L1 regularization", "lambda1", "[1e-6, 1e-1]", "Continuous (log scale)"], ["L2 regularization", "lambda2", "[1e-6, 1e-1]", "Continuous (log scale)"],
      ["Quantum rotation angle", "theta", "[0, 2 pi]", "Continuous"], ["Density parameter", "rho", "[0, 1]", "Continuous"]])
dump("table5", "Table 5", 17, ["setting", "value"],
     [["Encoder model dimension d_m", "128"], ["Bottleneck dimension d_b", "64"], ["Attention heads h", "8"], ["Key dimension d_k", "32"], ["Encoder dropout", "0.1"],
      ["QELM hidden layers L", "3"], ["QELM neurons per layer n_h", "512"], ["Activation", "GELU"], ["lambda1, lambda2", "3.2e-4, 1.7e-3"], ["theta", "1.84 rad"], ["rho", "0.63"],
      ["Phase 1 optimizer", "Adam, lr=1e-3, beta1=0.9, beta2=0.999"], ["Phase 2 solver", "Regularized Moore-Penrose pseudoinverse"], ["Loss", "Focal (gamma=2, alpha=0.25) + inverse-frequency class weights"],
      ["Batch size", "32"], ["Epochs", "100; early-stopping patience=15"], ["GWO population N", "20"], ["GWO budget T_max", "50; converged at 38 (multi-class) and 31 (binary)"],
      ["GWO fitness", "Five-fold CV macro-F1 on the training partition"], ["Hardware", "NVIDIA RTX 3080 (10 GB), Intel i7-10700, 32 GB RAM"],
      ["Framework", "Python 3.9, TensorFlow 2.11, scikit-learn 1.2, NumPy 1.24, pandas 1.5.3"], ["Random seeds", "{7, 13, 21, 42, 99}"],
      ["Code availability", "scripts will be released upon acceptance (Data availability section later names the upstream GitHub repository)"]])

# ---------------------------------------------------------------- Table 6 (binary)
t6 = [("Logistic Regression", 0.8821, 0.8794, 0.8612, 0.8341, 0.8474, 0.1073, 0.9214), ("Naive Bayes", 0.7634, 0.7601, 0.7289, 0.8923, 0.8025, 0.2187, 0.8734),
      ("Decision Tree", 0.9712, 0.9684, 0.9631, 0.9558, 0.9594, 0.0298, 0.9812), ("Random Forest", 0.9841, 0.9823, 0.9798, 0.9774, 0.9786, 0.0183, 0.9934),
      ("XGBoost", 0.9887, 0.9861, 0.9842, 0.9819, 0.9830, 0.0143, 0.9961), ("CNN-1D", 0.9891, 0.9874, 0.9856, 0.9831, 0.9843, 0.0138, 0.9968),
      ("LSTM", 0.9902, 0.9886, 0.9871, 0.9847, 0.9859, 0.0121, 0.9972), ("GRU", 0.9913, 0.9897, 0.9882, 0.9864, 0.9873, 0.0109, 0.9975),
      ("BiLSTM", 0.9924, 0.9911, 0.9896, 0.9879, 0.9887, 0.0096, 0.9979), ("DenseNet", 0.9931, 0.9918, 0.9904, 0.9887, 0.9895, 0.0088, 0.9981),
      ("Proposed Model", 0.9968, 0.9954, 0.9947, 0.9938, 0.9942, 0.0043, 0.9997)]
dump("table6", "Table 6", 18, ["model", "train_acc", "test_acc", "precision", "recall", "f1", "fpr", "auc_roc"], [list(r) for r in t6])

# ---------------------------------------------------------------- Table 7 (multiclass)
t7 = [("Logistic Regression", 0.7923, 0.7871, 0.7612, 0.7234, 0.7418, 0.1534, "3.1 min"), ("Naive Bayes", 0.6834, 0.6712, 0.6523, 0.7891, 0.7143, 0.2341, "0.8 min"),
      ("Decision Tree", 0.9541, 0.9494, 0.9412, 0.9387, 0.9399, 0.0381, "4.2 min"), ("Random Forest", 0.9712, 0.9676, 0.9631, 0.9584, 0.9607, 0.0241, "18.7 min"),
      ("XGBoost", 0.9783, 0.9741, 0.9698, 0.9653, 0.9675, 0.0198, "22.1 min"), ("CNN-1D", 0.9801, 0.9762, 0.9721, 0.9678, 0.9699, 0.0184, "34 min"),
      ("LSTM", 0.9823, 0.9789, 0.9748, 0.9712, 0.9730, 0.0162, "52 min"), ("GRU", 0.9841, 0.9807, 0.9771, 0.9738, 0.9754, 0.0147, "48 min"),
      ("BiLSTM", 0.9863, 0.9831, 0.9796, 0.9764, 0.9780, 0.0128, "61 min"), ("DenseNet", 0.9874, 0.9843, 0.9811, 0.9779, 0.9795, 0.0118, "58 min"),
      ("Proposed Model", 0.9921, 0.9894, 0.9871, 0.9848, 0.9859, 0.0079, "67 min")]
dump("table7", "Table 7", 19, ["model", "train_acc", "test_acc", "macro_precision", "macro_recall", "macro_f1", "fpr", "training_time"], [list(r) for r in t7])

# ---------------------------------------------------------------- Table 8 (per-class F1)
t8 = [("Normal", .9891, .9934, .9921, .9967, 0.33), ("DDoS_UDP", .9934, .9961, .9948, .9978, 0.17), ("DDoS_ICMP", .9912, .9943, .9931, .9971, 0.28),
      ("SQL_Injection", .9887, .9921, .9908, .9958, 0.37), ("DDoS_TCP", .9871, .9921, .9896, .9949, 0.28), ("DDoS_HTTP", .9756, .9914, .9812, .9891, -0.23),
      ("Password", .9712, .9834, .9774, .9867, 0.33), ("Vulnerability_Scanner", .9681, .9798, .9741, .9843, 0.45), ("Port_Scanning", .9643, .9764, .9712, .9824, 0.60),
      ("XSS", .9534, .9731, .9661, .9798, 0.67), ("Uploading", .9489, .9687, .9618, .9762, 0.75), ("Backdoor", .9412, .9641, .9554, .9714, 0.73),
      ("OS_Fingerprinting", .9287, .9587, .9461, .9673, 0.86), ("Ransomware", .9134, .9478, .9341, .9612, 1.34), ("MITM", .8723, .8764, .8698, .9421, 6.57)]
dump("table8", "Table 8", 20, ["class", "xgboost_f1", "densenet_f1", "bilstm_f1", "proposed_f1", "diff_vs_best_baseline_pp"], [list(r) for r in t8],
     ["only three comparators (XGBoost, DenseNet, BiLSTM) are printed; the other seven baselines have no per-class values in the article"])

# ---------------------------------------------------------------- Table 9 (five seeds)
t9 = [("Proposed Model", [.9847, .9866, .9854, .9871, .9858], .9859, .00095, [.9847, .9871], 1), ("DenseNet", [.9776, .9810, .9788, .9812, .9789], .9795, .00155, [.9776, .9814], 2),
      ("BiLSTM", [.9758, .9799, .9772, .9804, .9767], .9780, .00203, [.9755, .9805], 3), ("GRU", [.9731, .9778, .9748, .9779, .9734], .9754, .00233, [.9725, .9783], 4),
      ("LSTM", [.9704, .9755, .9724, .9757, .9710], .9730, .00248, [.9699, .9761], 5), ("CNN-1D", [.9669, .9728, .9692, .9728, .9678], .9699, .00277, [.9665, .9733], 6),
      ("XGBoost", [.9646, .9702, .9668, .9706, .9653], .9675, .00277, [.9641, .9709], 7), ("Random Forest", [.9572, .9636, .9598, .9641, .9588], .9607, .00303, [.9569, .9645], 8),
      ("Decision Tree", [.9357, .9434, .9389, .9442, .9373], .9399, .00375, [.9352, .9446], 9), ("Logistic Regression", [.7359, .7468, .7398, .7477, .7388], .7418, .00519, [.7354, .7482], 10),
      ("Naive Bayes", [.7081, .7197, .7124, .7209, .7104], .7143, .00570, [.7072, .7214], 11)]
dump("table9", "Table 9", 24, ["model", "runs_1_to_5", "mean_macro_f1", "sd", "ci95", "friedman_rank"], [list(r) for r in t9],
     ["Friedman chi2 = 50.00, df = 10, p = 2.67e-7 (text, Sec. 3.7); seeds {7, 13, 21, 42, 99} per Table 5"], friedman_printed={"chi2": 50.00, "df": 10, "p": 2.67e-7})

# ---------------------------------------------------------------- Table 10
t10 = [("DenseNet", .0064, .000145, 4.99), ("BiLSTM", .0079, .000211, 4.99), ("GRU", .0105, .000211, 5.92), ("LSTM", .0129, .000201, 6.87), ("CNN-1D", .0160, .000201, 7.73),
       ("XGBoost", .0184, .000145, 8.90), ("Random Forest", .0252, .000087, 11.24), ("Decision Tree", .0460, .000029, 16.84), ("Logistic Regression", .2441, 2.27e-7, 65.46),
       ("Naive Bayes", .2716, 2.27e-7, 66.46)]
dump("table10", "Table 10", 24, ["comparison_proposed_vs", "mean_diff_macro_f1", "holm_corrected_p", "cohens_d"], [list(r) for r in t10],
     ["the test variant and the effect-size denominator are not stated in the article"])

# ---------------------------------------------------------------- Table 11 (ablation)
t11 = [("Full proposed", .9894, .9871, .9848, .9859, .9421, .9612, .9673, None), ("w/o multi-head attention", .9821, .9783, .9742, .9762, .8967, .9381, .9420, -.0097),
       ("w/o QELM", .9856, .9821, .9797, .9809, .9183, .9489, .9534, -.0050), ("w/o quantum initialization", .9851, .9812, .9776, .9794, .9067, .9421, .9498, -.0065),
       ("w/o GWO", .9847, .9809, .9783, .9796, .9134, .9467, .9512, -.0063), ("w/o focal loss", .9830, .9827, .9701, .9763, .8794, .9357, .9382, -.0096),
       ("w/o GWO and w/o focal loss", .9791, .9742, .9618, .9679, .8512, .9189, .9237, -.0180), ("Encoder+MLP baseline", .9744, .9671, .9531, .9598, .7891, .8932, .9043, -.0261)]
dump("table11", "Table 11", 26, ["variant", "accuracy", "macro_p", "macro_r", "macro_f1", "mitm_f1", "ransomware_f1", "os_fingerprinting_f1", "delta_macro_f1"], [list(r) for r in t11])

# ---------------------------------------------------------------- Table 12
t12 = [("Logistic Regression", "735", .00147, "3.1", None, .012, None, None), ("Naive Bayes", "1,470", .00294, "0.8", None, .018, None, None), ("Decision Tree", "N/A (depth 18)", .00036, "4.2", None, .021, None, None),
       ("Random Forest (100 trees)", "N/A", .036, "18.7", None, .412, None, None), ("XGBoost (200 trees)", "N/A", .072, "22.1", None, .470, None, None), ("CNN-1D (3 conv blocks)", "0.34 M", 1.21, "34", .27, 3.84, 3704, 412),
       ("LSTM (2x128)", "0.58 M", 2.46, "52", .83, 12.71, 1205, 583), ("GRU (2x128)", "0.45 M", 1.93, "48", .71, 10.42, 1408, 524), ("BiLSTM (2x128)", "1.12 M", 4.87, "61", 1.47, 21.36, 680, 821),
       ("DenseNet-1D (4 blocks)", "1.84 M", 6.32, "58", .62, 8.91, 1613, 947), ("Proposed Attention-QELM-GWO", "0.21 M", 2.83, "67 (42 GWO + 25 training/solver)", .31, 4.12, 3226, 487)]
dump("table12", "Table 12", 27, ["model", "trainable_or_solved_params", "approx_ops_per_sample_M", "training_time_min", "gpu_ms_per_sample", "cpu_ms_per_sample", "gpu_samples_per_s", "peak_gpu_mb"],
     [list(r) for r in t12], ["'operations' vs FLOPs vs MACs is not defined in the article"])

# ---------------------------------------------------------------- text claims used by the audit
dump("text_claims", "text", 1, ["id", "quantity", "value", "where"],
     [["abstract_binary", "binary accuracy / F1 / AUC", [0.9954, 0.9942, 0.9997], "Abstract, Sec. 3.1"], ["abstract_multiclass", "15-class accuracy / macro-P / macro-R / macro-F1 / FPR", [0.9894, 0.9871, 0.9848, 0.9859, 0.0079], "Abstract, Sec. 3.2"],
      ["friedman_intro", "Friedman chi2 stated in contribution 5", 54.98, "Introduction, contribution 5 (page 4)"], ["friedman_results", "Friedman chi2 stated in results", 50.00, "Sec. 3.7 and Conclusion"],
      ["cohen_d_intro", "Cohen's d range stated in contribution 5", [1.87, 14.38], "Introduction, contribution 5 (page 4)"], ["cohen_d_table10", "Cohen's d range printed in Table 10", [4.99, 66.46], "Table 10"],
      ["fp_binary_proposed", "approx. false alarms out of 273,592 benign flows", 1176, "Sec. 3.1"], ["fp_binary_densenet", "false alarms for DenseNet", 2408, "Sec. 3.1"], ["fp_binary_xgboost", "false alarms for XGBoost", 3911, "Sec. 3.1"],
      ["imbalance_ratio", "Normal : MITM", "> 788 : 1", "Sec. 2.1"], ["total_records", "records", 1909671, "Abstract, Sec. 2.1"], ["gwo_iterations", "iterations to convergence (multi-class, binary)", [38, 31], "Sec. 2.5.1"],
      ["gwo_budget", "fitness evaluations per classification mode", 1000, "Sec. 2.5.1 (N=20, T_max=50)"], ["ablation_text_deltas", "macro-F1 deltas in text", {"attention": 0.0097, "focal": 0.0096, "qelm": 0.0050, "quantum_init": 0.0065, "gwo": 0.0063, "gwo_and_focal": 0.0180, "encoder_mlp": 0.0261}, "Sec. 3.8"],
      ["table12_text", "full model cost", {"params_M": 0.21, "ops_M": 2.83, "gpu_ms": 0.31, "cpu_ms": 4.12, "gpu_samples_per_s": 3226, "peak_gpu_mb": 487, "train_min": 67, "gwo_min": 42, "fit_min": 25}, "Sec. 3.8"]])
print("wrote", len(list(OUT.glob('*.json'))), "files to", OUT)
