#!/usr/bin/env python3

'''
## Model building: normal vs cancer.

Methylation + protein. Elastic-net logistic regression, nested patient-grouped
CV. All data-driven steps (median imputation of methylation, log transform and
blank imputation at min / sqrt(2) of protein, variance filter, correlation
filter, scaling) live in the Pipeline, so they are fitted on each training
fold only. Feature selection is the elastic-net L1 penalty, tuned in the inner
CV.

'''

import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from classifer_4_data_preprocessing_meth import make_meth_imputer
from classifer_4_data_preprocessing_prot import make_log_transformer, make_prot_imputer
from classifer_5_logireg_meth_only import (
    CORR_THRESHOLD, N_SPLITS, PARAM_GRID, SEED,
    CorrelationFilter, loss_curves, plot_training_loss)

VAR_THRESHOLD = 1e-4  # one threshold for both blocks; smallest log-protein variance is ~0.08


def make_pipeline(meth_cols, prot_cols) -> Pipeline:
    '''
    Columns are concatenated methylation first, then protein, so in the
    correlation filter a methylation feature is kept over a correlated protein.
    '''

    prep = ColumnTransformer([
        ("meth", make_meth_imputer(), meth_cols),
        ("prot", Pipeline([("log", make_log_transformer()),
                           ("imp", make_prot_imputer("div_sqrt2"))]), prot_cols),
    ])

    return Pipeline([
        ("prep", prep),
        ("var", VarianceThreshold(VAR_THRESHOLD)),
        ("corr", CorrelationFilter(CORR_THRESHOLD)),
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(penalty="elasticnet", solver="saga",
                                   class_weight="balanced", max_iter=5000,
                                   random_state=SEED)),
    ])


def logistic_regression(df_samples, df_meth, df_prot, out_dir: Path):

    df = (df_samples.merge(df_meth, on="sample_id", how="inner")
                    .merge(df_prot, on="sample_id", how="inner"))
    meth_cols = [c for c in df_meth.columns if c != "sample_id"]
    prot_cols = [c for c in df_prot.columns if c != "sample_id"]
    X = df[meth_cols + prot_cols]
    y = (df["cancer_status"] == "cancer").astype(int).to_numpy()
    groups = df["patient_id"].to_numpy()
    is_meth = np.r_[np.ones(len(meth_cols), bool), np.zeros(len(prot_cols), bool)]

    outer = StratifiedGroupKFold(N_SPLITS, shuffle=True, random_state=SEED)
    rows, curves, cv_tables = [], [], []

    for k, (tr, te) in enumerate(outer.split(X, y, groups), 1):
        assert not set(groups[tr]) & set(groups[te])

        inner = StratifiedGroupKFold(N_SPLITS, shuffle=True, random_state=SEED)
        gs = GridSearchCV(make_pipeline(meth_cols, prot_cols), PARAM_GRID,
                          scoring="neg_log_loss", cv=inner, n_jobs=-1)
        gs.fit(X.iloc[tr], y[tr], groups=groups[tr])
        best = gs.best_estimator_

        # map the surviving, selected features back to modality
        kept = is_meth[best["var"].get_support()][best["corr"].keep_]
        sel = best["clf"].coef_.ravel() != 0

        proba = best.predict_proba(X.iloc[te])[:, 1]
        rows.append({
            "fold": k,
            "start": X.shape[1],
            "after_var": int(best["var"].get_support().sum()),
            "after_corr": int(best["corr"].keep_.sum()),
            "selected": int(sel.sum()),
            "sel_meth": int((sel & kept).sum()),
            "sel_prot": int((sel & ~kept).sum()),
            "C": best["clf"].C,
            "l1_ratio": best["clf"].l1_ratio,
            "auc": roc_auc_score(y[te], proba),
            "bacc": balanced_accuracy_score(y[te], proba > 0.5),
        })

        curves.append(loss_curves(best, X.iloc[tr], y[tr], X.iloc[te], y[te]))
        cv_tables.append(pd.DataFrame(gs.cv_results_))

    res = pd.DataFrame(rows).set_index("fold")
    print(res.to_string(float_format=lambda v: f"{v:.3f}"))
    print(f"\nFeatures: {X.shape[1]} ({len(meth_cols)} meth + {len(prot_cols)} prot)"
          f" -> VarianceThreshold {res['after_var'].mean():.1f}"
          f" -> CorrelationFilter {res['after_corr'].mean():.1f}"
          f" -> selected {res['selected'].mean():.1f} (mean over folds)")
    print(f"AUC  {res['auc'].mean():.3f} +/- {res['auc'].std():.3f}")
    print(f"BACC {res['bacc'].mean():.3f} +/- {res['bacc'].std():.3f}")

    plot_training_loss(curves, cv_tables, Path(out_dir, "logireg_meth_prot_loss.png"))

    return res


def main() -> int:

    pwd = Path(__file__).parent

    df_samples = pd.read_csv(Path(pwd, "data/train_samples.csv"))

    df_meth = pd.read_csv(Path(pwd, "data/train_methylation.csv"))

    df_prot = pd.read_csv(Path(pwd, "data/train_protein.csv"))

    logistic_regression(df_samples, df_meth, df_prot, Path(pwd, "data"))

    return 0


if __name__ == '__main__':
    sys.exit(main())
