from .classical import CLASSICAL, SklearnAdapter  # noqa: F401
from .neural import NEURAL, TorchAdapter  # noqa: F401

ALL_BASELINES = CLASSICAL + NEURAL          # the paper's ten comparators
LABELS = {"logistic_regression": "Logistic Regression", "naive_bayes": "Naive Bayes", "decision_tree": "Decision Tree", "random_forest": "Random Forest",
          "xgboost": "XGBoost", "cnn1d": "CNN-1D", "lstm": "LSTM", "gru": "GRU", "bilstm": "BiLSTM", "densenet1d": "DenseNet"}


def make_baseline(name: str, n_classes: int, seed: int = 0, **kw):
    if name in CLASSICAL:
        return SklearnAdapter(name, n_classes, seed, **{k: v for k, v in kw.items() if k in ("weighted", "n_jobs")})
    if name in NEURAL:
        return TorchAdapter(name, n_classes, seed, **{k: v for k, v in kw.items() if k in ("weighted", "train")})
    raise KeyError(f"unknown baseline {name!r}; choose from {ALL_BASELINES}")
