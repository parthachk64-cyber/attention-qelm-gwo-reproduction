"""Classical baselines on the SAME split rows, class mapping and score orientation as the proposed model.

Settings follow the article / upstream where given: decision tree max_depth 18, random forest 100 trees, XGBoost 200 trees
(depth 8, lr 0.1, hist), Gaussian naive Bayes, logistic regression. Deviations (documented in docs/reproduction.md):
logistic regression uses lbfgs (upstream: saga, 2000 iterations, which is impractically slow at 1.5 M rows).
Class weighting: LR/DT/RF use class_weight='balanced'; XGBoost and Gaussian NB receive balanced per-row sample weights
(the upstream XGBoost call applies none, NB has no class-weight argument). ``weighted=False`` disables weighting everywhere."""
from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_sample_weight

from ..errors import MissingInputError

CLASSICAL = ("logistic_regression", "naive_bayes", "decision_tree", "random_forest", "xgboost")


class SklearnAdapter:
    kind = "classical"

    def __init__(self, name: str, n_classes: int, seed: int = 0, weighted: bool = True, n_jobs: int = 1, **overrides):
        self.name, self.n_classes, self.seed, self.weighted, self.overrides = name, n_classes, seed, weighted, overrides
        cw = "balanced" if weighted else None
        if name == "logistic_regression":
            self.model = LogisticRegression(solver="lbfgs", max_iter=overrides.get("max_iter", 500), class_weight=cw, random_state=seed)
            self.use_sample_weight = False
        elif name == "naive_bayes":
            self.model, self.use_sample_weight = GaussianNB(), weighted
        elif name == "decision_tree":
            self.model = DecisionTreeClassifier(max_depth=overrides.get("max_depth", 18), class_weight=cw, random_state=seed)
            self.use_sample_weight = False
        elif name == "random_forest":
            self.model = RandomForestClassifier(n_estimators=overrides.get("n_estimators", 100), class_weight=cw, n_jobs=n_jobs, random_state=seed)
            self.use_sample_weight = False
        elif name == "xgboost":
            try:
                from xgboost import XGBClassifier
            except ImportError as e:
                raise MissingInputError("the xgboost package", "pip install 'iiot-qelm[xgboost]'") from e
            self.model = XGBClassifier(n_estimators=overrides.get("n_estimators", 200), max_depth=overrides.get("max_depth", 8), learning_rate=0.1,
                                       tree_method="hist", n_jobs=n_jobs, random_state=seed, verbosity=0)
            self.use_sample_weight = weighted
        else:
            raise KeyError(f"unknown classical baseline {name!r}; choose from {CLASSICAL}")
        self.history: dict = {}

    def fit(self, X, y, class_weights=None, **_):
        self.classes_ = np.unique(y)
        sw = compute_sample_weight("balanced", y) if self.use_sample_weight else None
        self.model.fit(X, y, **({"sample_weight": sw} if sw is not None else {}))
        return self

    def decision_scores(self, X) -> np.ndarray:
        p = self.model.predict_proba(X)
        if p.shape[1] != self.n_classes:                           # a class absent from training: pad with zeros
            full = np.zeros((len(X), self.n_classes))
            full[:, self.classes_] = p
            p = full
        return p

    def predict(self, X):
        return self.decision_scores(X).argmax(axis=1)

    def n_params(self) -> Optional[int]:
        if self.name == "logistic_regression":
            return int(self.model.coef_.size + self.model.intercept_.size)
        if self.name == "naive_bayes":
            return int(self.model.theta_.size + self.model.var_.size + self.model.class_prior_.size)
        return None                                                # tree ensembles: Table 12 prints N/A
