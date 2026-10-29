#!/usr/bin/env python3

'''
## Feature selection: normal vs cancer.

Protein only. Stability selection with L1 logistic regression on
patient-grouped, class-stratified complementary-pair subsamples. Run once per
outer validation fold (training portion only) and once on all training data.
No model is fitted on the selected features.

All data-driven steps (blank imputation at min / sqrt(2) per marker, variance
filter, correlation filter, scaling) live in the Pipeline, so they are fitted
on each subsample only.

'''

import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from itertools import combinations
from joblib import Parallel, delayed
from pathlib import Path
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from classifer_4_data_preprocessing_prot import make_log_transformer, make_prot_imputer
from classifer_5_logireg_meth_only import CORR_THRESHOLD, N_SPLITS, SEED, CorrelationFilter

VAR_THRESHOLD = 1e-2  # log-scale variance; the smallest marker is ~0.08
N_PAIRS = 100       # complementary pairs -> 2 * N_PAIRS subsamples of 50%
N_SELECT = 10
TARGET_Q = 10       # average nonzero coefficients per subsample fit
C_BOUNDS = (-3.0, 1.0)  # log10 search range for C
C_ITERS = 8
C_PAIRS = 3


def selection_pipeline(C: float) -> Pipeline:

    return Pipeline([
        ("log", make_log_transformer()),
        ("imp", make_prot_imputer("div_sqrt2")),
        ("var", VarianceThreshold(VAR_THRESHOLD)),
        ("corr", CorrelationFilter(CORR_THRESHOLD)),
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(penalty="l1", solver="saga", C=C,
                                   class_weight="balanced", max_iter=5000,
                                   random_state=SEED)),
    ])


def fit_coefs(X: pd.DataFrame, y, C: float) -> np.ndarray:
    '''
    Fit the L1 pipeline and return one coefficient per original column
    (0 for features dropped by the variance / correlation filters).
    '''

    pipe = selection_pipeline(C)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        pipe.fit(X, y)

    coefs = np.zeros(X.shape[1])
    kept = np.flatnonzero(pipe["var"].get_support())[pipe["corr"].keep_]
    coefs[kept] = pipe["clf"].coef_.ravel()

    return coefs


def pair_halves(y, groups, b: int):
    '''
    Disjoint, patient-grouped, class-stratified halves for pair b.
    '''

    cv = StratifiedGroupKFold(2, shuffle=True, random_state=SEED + b)
    h1, h2 = next(cv.split(np.zeros(len(y)), y, groups))
    assert not set(groups[h1]) & set(groups[h2])

    return h1, h2


def pick_C_for_q(X: pd.DataFrame, y, groups) -> float:
    '''
    Bisection on log10(C) so the mean number of nonzero coefficients over a
    few half-samples is ~TARGET_Q (nonzero count grows with C).
    '''

    halves = [h for b in range(C_PAIRS) for h in pair_halves(y, groups, b)]
    lo, hi = C_BOUNDS

    for _ in range(C_ITERS):
        mid = (lo + hi) / 2
        q = np.mean([(fit_coefs(X.iloc[h], y[h], 10 ** mid) != 0).sum()
                     for h in halves])
        if q < TARGET_Q:
            lo = mid
        else:
            hi = mid

    return 10 ** ((lo + hi) / 2)


def stability_selection(X: pd.DataFrame, y, groups) -> tuple[pd.DataFrame, float, float]:
    '''
    Selection frequency of each feature over 2 * N_PAIRS half-samples, ranked
    by frequency then mean |coef| among the fits that selected it.
    '''

    C = pick_C_for_q(X, y, groups)

    halves = []
    for b in range(N_PAIRS):
        h1, h2 = pair_halves(y, groups, b)
        assert min(y[h1].mean(), y[h2].mean()) > 0.3 and max(y[h1].mean(), y[h2].mean()) < 0.7
        halves += [h1, h2]

    coefs = np.array(Parallel(n_jobs=-1)(
        delayed(fit_coefs)(X.iloc[h], y[h], C) for h in halves))

    sel = coefs != 0
    n_sel = sel.sum(axis=0)
    res = pd.DataFrame({
        "freq": n_sel / len(halves),
        "mean_abs_coef": np.abs(coefs).sum(axis=0) / np.maximum(n_sel, 1),
    }, index=X.columns)
    res = res.sort_values(["freq", "mean_abs_coef"], ascending=False)

    return res, C, sel.sum(axis=1).mean()


def plot_stability(freq: pd.Series, out_path: Path) -> None:

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(np.arange(1, len(freq) + 1), freq.to_numpy(), marker=".", lw=1)
    ax.axvline(N_SELECT + 0.5, color="tab:red", ls="--", lw=1,
               label=f"top {N_SELECT} cut")
    ax.set(xlabel="feature rank", ylabel="selection frequency",
           title="Stability selection (all training data)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def stability_workflow(df_samples, df_prot, out_dir: Path):

    df = df_samples.merge(df_prot, on="sample_id", how="inner")
    feats = [c for c in df_prot.columns if c != "sample_id"]
    X = df[feats]
    y = (df["cancer_status"] == "cancer").astype(int).to_numpy()
    groups = df["patient_id"].to_numpy()

    outer = StratifiedGroupKFold(N_SPLITS, shuffle=True, random_state=SEED)
    cands, fold_freq = [], {}

    for k, (tr, te) in enumerate(outer.split(X, y, groups), 1):
        assert not set(groups[tr]) & set(groups[te])

        res, C, q = stability_selection(X.iloc[tr], y[tr], groups[tr])
        top = res.head(N_SELECT)
        print(f"fold {k}: C={C:.4g}  mean nonzero={q:.1f}  "
              f"freq@{N_SELECT}={res['freq'].iloc[N_SELECT - 1]:.2f}  "
              f"freq@{N_SELECT + 1}={res['freq'].iloc[N_SELECT]:.2f}")

        cands.append(top.reset_index(names="feature").assign(
            fold=k, rank=np.arange(1, N_SELECT + 1)))
        fold_freq[f"fold_{k}"] = res["freq"]

    cands = pd.concat(cands)[["fold", "rank", "feature", "freq", "mean_abs_coef"]]
    cands.to_csv(Path(out_dir, "prot_stability_fold_candidates.csv"), index=False)
    pd.DataFrame(fold_freq).reindex(feats).to_csv(
        Path(out_dir, "prot_stability_fold_freq.csv"), index_label="feature")

    res, C, q = stability_selection(X, y, groups)
    print(f"final: C={C:.4g}  mean nonzero={q:.1f}  "
          f"freq@{N_SELECT}={res['freq'].iloc[N_SELECT - 1]:.2f}  "
          f"freq@{N_SELECT + 1}={res['freq'].iloc[N_SELECT]:.2f}")

    res.reset_index(names="feature").to_csv(
        Path(out_dir, "prot_stability_final_freq.csv"), index=False)

    final = res.head(N_SELECT).reset_index(names="feature")
    final.insert(0, "rank", np.arange(1, N_SELECT + 1))
    counts = cands["feature"].value_counts()
    final["n_folds_selected"] = final["feature"].map(counts).fillna(0).astype(int)
    final.to_csv(Path(out_dir, f"prot_stability_final_top{N_SELECT}.csv"), index=False)

    plot_stability(res["freq"], Path(out_dir, "prot_stability_freq.png"))

    sets = {k: set(g["feature"]) for k, g in cands.groupby("fold")}
    jac = [len(sets[a] & sets[b]) / len(sets[a] | sets[b])
           for a, b in combinations(sets, 2)]
    fin = set(final["feature"])
    print(f"\nfold top-{N_SELECT} pairwise Jaccard: mean {np.mean(jac):.2f}, "
          f"min {np.min(jac):.2f}")
    print("overlap with final list: " +
          ", ".join(f"fold {k}={len(s & fin)}" for k, s in sets.items()))
    print("\n" + final.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    return final


def main() -> int:

    pwd = Path(__file__).parent

    df_samples = pd.read_csv(Path(pwd, "data/train_samples.csv"))

    df_prot = pd.read_csv(Path(pwd, "data/train_protein.csv"))

    stability_workflow(df_samples, df_prot, Path(pwd, "data"))

    return 0


if __name__ == '__main__':
    sys.exit(main())
