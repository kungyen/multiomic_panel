#!/usr/bin/env python3

'''
## Final model: refit on the full training set, evaluate on the held-out test set.

Elastic-net logistic regression on the final exhaustive panel (panel_final.json).
Same imputation (median for methylation; log + blank imputation at min / sqrt(2)
for protein) and scaling as classifer_5_logireg_meth_prot.py. The variance and
correlation filters are left out: they screen a large feature space and could
only drop members of an already fixed panel. C and l1_ratio are tuned once by
patient-grouped CV on the whole training set. Every data-driven step is fitted
on train only and then applied unchanged to the test set, which is scored once.

'''

import json
import sys
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (balanced_accuracy_score, confusion_matrix, f1_score,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from classifer_4_data_preprocessing_meth import make_meth_imputer
from classifer_4_data_preprocessing_prot import (
    load_prot_features, make_log_transformer, make_prot_imputer)
from classifer_5_logireg_meth_only import (
    LOSS_ITERS, N_SPLITS, PARAM_GRID, SEED, loss_curves)

PANEL_METHOD = "exhaustive"
THRESHOLD = 0.5
N_BOOT = 2000


def make_pipeline(meth_cols, prot_cols) -> Pipeline:
    '''Same steps as classifer_5_logireg_meth_prot.make_pipeline, minus the filters.'''

    prep = ColumnTransformer([
        ("meth", make_meth_imputer(), meth_cols),
        ("prot", Pipeline([("log", make_log_transformer()),
                           ("imp", make_prot_imputer("div_sqrt2"))]), prot_cols),
    ])

    return Pipeline([
        ("prep", prep),
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(penalty="elasticnet", solver="saga",
                                   class_weight="balanced", max_iter=5000,
                                   random_state=SEED)),
    ])


def load_split(df_samples, df_meth, df_prot, cols):
    '''Samples with panel features, rows in the order of df_samples.'''

    df = (df_samples.merge(df_meth, on="sample_id", how="left")
                    .merge(df_prot.reset_index(), on="sample_id", how="left"))
    assert len(df) == len(df_samples) and not df[cols].isna().all(axis=1).any()
    y = (df["cancer_status"] == "cancer").astype(int).to_numpy()

    return df, df[cols], y


def metrics(y, proba) -> dict:

    pred = proba > THRESHOLD
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()

    return {
        "auc": roc_auc_score(y, proba),
        "sensitivity": tp / (tp + fn),
        "specificity": tn / (tn + fp),
        "f1": f1_score(y, pred),
        "bacc": balanced_accuracy_score(y, pred),
        "tp": tp, "fn": fn, "tn": tn, "fp": fp,
    }


def bootstrap_auc_ci(y, proba, groups):
    '''Percentile 95% CI of the AUC, resampling patients (a patient can have several samples).'''

    rng = np.random.default_rng(SEED)
    idx = pd.Series(np.arange(len(y))).groupby(groups).apply(np.array)
    aucs = []
    for _ in range(N_BOOT):
        take = np.concatenate(idx.sample(len(idx), replace=True,
                                         random_state=int(rng.integers(2**31))).to_list())
        if len(set(y[take])) == 2:
            aucs.append(roc_auc_score(y[take], proba[take]))

    return np.percentile(aucs, [2.5, 97.5])


def plot_refit_loss(train_curve, cv_table, out_path: Path) -> None:
    '''Training objective per saga iteration, plus the regularization path.'''

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    ax1.plot(np.arange(1, LOSS_ITERS + 1), train_curve, color="tab:blue", lw=2.5)
    ax1.set(xlabel="saga iteration", ylabel="loss",
            title="Loss during refit on full training set\n(log-loss + penalty)")

    cv = cv_table.groupby(["param_clf__l1_ratio", "param_clf__C"])[
        "mean_test_score"].mean().mul(-1).reset_index()
    for l1r, g in cv.groupby("param_clf__l1_ratio"):
        ax2.plot(g["param_clf__C"], g["mean_test_score"], marker="o",
                 label=f"l1_ratio={l1r}")
    ax2.set(xscale="log", xlabel="C (inverse regularization)",
            ylabel="grouped-CV log-loss", title="Regularization path (training set)")
    ax2.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_roc(y, proba, auc, out_path: Path) -> None:

    fpr, tpr, _ = roc_curve(y, proba)
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.plot(fpr, tpr, lw=2, label=f"test AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], color="grey", ls="--", lw=1)
    ax.set(xlabel="1 - specificity", ylabel="sensitivity", title="Held-out test ROC")
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def refit_and_test(train, test, panel, out_dir: Path) -> dict:

    meth_cols, prot_cols = panel["methylation"], panel["protein"]
    cols = meth_cols + prot_cols

    tr_samples, df_meth, df_prot = train
    df_tr, X_tr, y_tr = load_split(tr_samples, df_meth, df_prot, cols)
    te_samples, df_meth, df_prot = test
    df_te, X_te, y_te = load_split(te_samples, df_meth, df_prot, cols)

    g_tr, g_te = df_tr["patient_id"].to_numpy(), df_te["patient_id"].to_numpy()
    assert not set(g_tr) & set(g_te)

    # tune on the training set only, then refit on all of it
    cv = StratifiedGroupKFold(N_SPLITS, shuffle=True, random_state=SEED)
    gs = GridSearchCV(make_pipeline(meth_cols, prot_cols), PARAM_GRID,
                      scoring="neg_log_loss", cv=cv, n_jobs=-1, refit=True)
    gs.fit(X_tr, y_tr, groups=g_tr)
    model = gs.best_estimator_
    C, l1r = model["clf"].C, model["clf"].l1_ratio
    print(f"Panel ({PANEL_METHOD}): {len(meth_cols)} meth + {len(prot_cols)} prot,"
          f" train n={len(y_tr)}, test n={len(y_te)}")
    print(f"Best C={C:g}, l1_ratio={l1r:g}")
    print(pd.Series(model["clf"].coef_.ravel(), index=cols, name="coef")
            .to_string(float_format=lambda v: f"{v:.3f}"))

    # preprocessed (imputed / log-transformed, unscaled) panel data; test uses train-fitted steps
    for name, df, X, y in (("train", df_tr, X_tr, y_tr), ("test", df_te, X_te, y_te)):
        out = df[["sample_id", "patient_id", "cancer_status"]].copy()
        out[cols] = model["prep"].transform(X)
        out.to_csv(Path(out_dir, f"{name}_panel.csv"), index=False)

    # loss per iteration while refitting (train data only; the test set is not touched)
    train_curve, _ = loss_curves(model, X_tr, y_tr, X_tr, y_tr)
    plot_refit_loss(train_curve, pd.DataFrame(gs.cv_results_),
                    Path(out_dir, "logireg_final_loss.png"))

    # the one look at the test set
    proba = model.predict_proba(X_te)[:, 1]
    res = metrics(y_te, proba)
    lo, hi = bootstrap_auc_ci(y_te, proba, g_te)
    res.update({"auc_ci_low": lo, "auc_ci_high": hi, "C": C, "l1_ratio": l1r,
                "n_train": len(y_tr), "n_test": len(y_te)})

    print(f"\nTest AUC {res['auc']:.3f} (95% CI {lo:.3f}-{hi:.3f})")
    print(f"Sensitivity {res['sensitivity']:.3f}  Specificity {res['specificity']:.3f}"
          f"  F1 {res['f1']:.3f}  BACC {res['bacc']:.3f}  (threshold {THRESHOLD})")
    print(f"TP {res['tp']}  FN {res['fn']}  TN {res['tn']}  FP {res['fp']}")

    joblib.dump(model, Path(out_dir, "model_final.joblib"))
    pd.Series(res, name="value").to_csv(Path(out_dir, "test_metrics.csv"), index_label="metric")
    pd.DataFrame({"sample_id": df_te["sample_id"], "patient_id": g_te,
                  "y_true": y_te, "proba_cancer": proba}
                 ).to_csv(Path(out_dir, "test_predictions.csv"), index=False)
    plot_roc(y_te, proba, res["auc"], Path(out_dir, "test_roc.png"))

    return res


def main() -> int:

    pwd = Path(__file__).parent
    data = Path(pwd, "data")

    panel = json.loads(Path(data, "panel_final.json").read_text())[PANEL_METHOD]

    train = (pd.read_csv(Path(data, "train_samples.csv")),
             pd.read_csv(Path(data, "train_methylation.csv")),
             pd.read_csv(Path(data, "train_protein.csv")).set_index("sample_id"))

    # no test feature files: take the test samples out of the full tables
    te_samples = pd.read_csv(Path(data, "test_samples.csv"))
    ids = set(te_samples["sample_id"])
    df_meth = pd.read_csv(Path(data, "methylation.csv"))
    df_prot = load_prot_features(pd.read_csv(Path(data, "protein.csv")))
    test = (te_samples,
            df_meth[df_meth["sample_id"].isin(ids)],
            df_prot[df_prot.index.isin(ids)])

    refit_and_test(train, test, panel, data)

    return 0


if __name__ == '__main__':
    sys.exit(main())
