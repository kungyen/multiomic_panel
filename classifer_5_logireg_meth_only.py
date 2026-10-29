#!/usr/bin/env python3

'''
## Model building: normal vs cancer.

Methylation only. Elastic-net logistic regression, nested patient-grouped CV.
All data-driven steps (median imputation, variance filter, correlation
filter, scaling) live in the Pipeline, so they are fitted on each training
fold only.

'''

import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

from classifer_4_data_preprocessing_meth import make_meth_imputer

VAR_THRESHOLD = 1e-4
CORR_THRESHOLD = 0.9
N_SPLITS = 5
SEED = 0
PARAM_GRID = {
    "clf__C": np.logspace(-3, 1, 5),
    "clf__l1_ratio": [0.1, 0.5, 0.9],
}
LOSS_ITERS = 200


class CorrelationFilter(BaseEstimator, TransformerMixin):
    '''
    Drop one feature of every pair with |Pearson r| > threshold. Features are
    scanned in column order; the first of a correlated pair is kept.
    '''

    def __init__(self, threshold=CORR_THRESHOLD):
        self.threshold = threshold

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        corr = np.abs(np.corrcoef(X, rowvar=False))
        keep = np.ones(X.shape[1], dtype=bool)
        for i in range(X.shape[1]):
            if keep[i]:
                drop = corr[i, i + 1:] > self.threshold
                keep[i + 1:] &= ~drop
        self.keep_ = keep
        return self

    def transform(self, X):
        return np.asarray(X, dtype=float)[:, self.keep_]


def make_pipeline() -> Pipeline:

    return Pipeline([
        ("imp", make_meth_imputer()),
        ("var", VarianceThreshold(VAR_THRESHOLD)),
        ("corr", CorrelationFilter(CORR_THRESHOLD)),
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(penalty="elasticnet", solver="saga",
                                   class_weight="balanced", max_iter=5000,
                                   random_state=SEED)),
    ])


def loss_curves(best: Pipeline, X_tr, y_tr, X_te, y_te):
    '''
    Per-iteration training objective and held-out log-loss. saga exposes no
    loss history, so refit the tuned classifier with warm_start one iteration
    at a time on the already-transformed fold data.
    '''

    Z_tr, Z_te = best[:-1].transform(X_tr), best[:-1].transform(X_te)
    p = best["clf"].get_params()
    clf = LogisticRegression(penalty="elasticnet", solver="saga", C=p["C"],
                             l1_ratio=p["l1_ratio"], class_weight="balanced",
                             warm_start=True, max_iter=1, random_state=SEED)
    w = compute_sample_weight("balanced", y_tr)
    n, l1r = len(y_tr), p["l1_ratio"]
    train, test = [], []

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        for _ in range(LOSS_ITERS):
            clf.fit(Z_tr, y_tr)
            wv = clf.coef_.ravel()
            pen = (l1r * np.abs(wv).sum() + 0.5 * (1 - l1r) * (wv ** 2).sum()) / (p["C"] * n)
            train.append(log_loss(y_tr, clf.predict_proba(Z_tr)[:, 1],
                                  sample_weight=w, labels=[0, 1]) + pen)
            test.append(log_loss(y_te, clf.predict_proba(Z_te)[:, 1], labels=[0, 1]))

    return np.array(train), np.array(test)


def plot_training_loss(curves, cv_tables, out_path: Path) -> None:

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    it = np.arange(1, LOSS_ITERS + 1)

    for tr, te in curves:
        ax1.plot(it, tr, color="tab:blue", alpha=0.3, lw=1)
        ax1.plot(it, te, color="tab:orange", alpha=0.3, lw=1, ls="--")
    ax1.plot(it, np.mean([c[0] for c in curves], axis=0), color="tab:blue",
             lw=2.5, label="train objective (log-loss + penalty)")
    ax1.plot(it, np.mean([c[1] for c in curves], axis=0), color="tab:orange",
             lw=2.5, ls="--", label="held-out log-loss")
    ax1.set(xlabel="saga iteration", ylabel="loss",
            title="Loss during training (thin = outer folds, bold = mean)")
    ax1.legend(frameon=False)

    cv = pd.concat(cv_tables).groupby(["param_clf__l1_ratio", "param_clf__C"])[
        "mean_test_score"].mean().mul(-1).reset_index()
    for l1r, g in cv.groupby("param_clf__l1_ratio"):
        ax2.plot(g["param_clf__C"], g["mean_test_score"], marker="o",
                 label=f"l1_ratio={l1r}")
    ax2.set(xscale="log", xlabel="C (inverse regularization)",
            ylabel="inner-CV log-loss", title="Regularization path")
    ax2.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def logistic_regression(df_samples, df_meth, out_dir: Path):

    df = df_samples.merge(df_meth, on="sample_id", how="inner")
    feats = [c for c in df_meth.columns if c != "sample_id"]
    X = df[feats]
    y = (df["cancer_status"] == "cancer").astype(int).to_numpy()
    groups = df["patient_id"].to_numpy()

    outer = StratifiedGroupKFold(N_SPLITS, shuffle=True, random_state=SEED)
    rows, curves, cv_tables = [], [], []

    for k, (tr, te) in enumerate(outer.split(X, y, groups), 1):
        assert not set(groups[tr]) & set(groups[te])

        inner = StratifiedGroupKFold(N_SPLITS, shuffle=True, random_state=SEED)
        gs = GridSearchCV(make_pipeline(), PARAM_GRID, scoring="neg_log_loss",
                          cv=inner, n_jobs=-1)
        gs.fit(X.iloc[tr], y[tr], groups=groups[tr])
        best = gs.best_estimator_

        proba = best.predict_proba(X.iloc[te])[:, 1]
        rows.append({
            "fold": k,
            "start": len(feats),
            "after_var": int(best["var"].get_support().sum()),
            "after_corr": int(best["corr"].keep_.sum()),
            "nonzero_coef": int((best["clf"].coef_ != 0).sum()),
            "C": best["clf"].C,
            "l1_ratio": best["clf"].l1_ratio,
            "auc": roc_auc_score(y[te], proba),
            "bacc": balanced_accuracy_score(y[te], proba > 0.5),
        })

        curves.append(loss_curves(best, X.iloc[tr], y[tr], X.iloc[te], y[te]))
        cv_tables.append(pd.DataFrame(gs.cv_results_))

    res = pd.DataFrame(rows).set_index("fold")
    print(res.to_string(float_format=lambda v: f"{v:.3f}"))
    print(f"\nAUC  {res['auc'].mean():.3f} +/- {res['auc'].std():.3f}")
    print(f"BACC {res['bacc'].mean():.3f} +/- {res['bacc'].std():.3f}")

    plot_training_loss(curves, cv_tables, Path(out_dir, "logireg_meth_loss.png"))

    return res


def main() -> int:

    pwd = Path(__file__).parent

    df_samples = pd.read_csv(Path(pwd, "data/train_samples.csv"))

    df_meth = pd.read_csv(Path(pwd, "data/train_methylation.csv"))

    logistic_regression(df_samples, df_meth, Path(pwd, "data"))

    return 0


if __name__ == '__main__':
    sys.exit(main())
