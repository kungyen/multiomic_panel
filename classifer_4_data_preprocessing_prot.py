#!/usr/bin/env python3

'''
## Protein preprocessing.

Features are the 40 protein markers (sample_id, plate_id and well excluded),
log-transformed, imputed and standardized. Blanks in protein.csv mean "below the lower limit of
quantitation (LLOQ)", i.e. left-censored (MNAR) with an unknown limit. The
LLOQ of each marker is proxied by the smallest observed value of that marker
in the training fold, and blanks are replaced by a value below it
(see LloqImputer). Strategies are compared by CV and by a simulated-censoring
fidelity check.

'''

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import minimize
from scipy.stats import norm
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

PROT_EXCLUDE = ["sample_id", "plate_id", "well"]


class LloqImputer(BaseEstimator, TransformerMixin):
    '''
    Replace blanks (NaN, below LLOQ) in log-scale data by a value below the
    per-column LLOQ proxy (column minimum seen in fit).

    strategy="div":   fill = log(lloq / divisor)     (sqrt(2) and 2 are the usual choices)
    strategy="tobit": fill = E[x | x < log(lloq)] under a left-censored
                      normal fitted per column by maximum likelihood.
    lloq_scale rescales the LLOQ proxy (sensitivity analysis, 1.0 = column min).
    add_indicator appends a 0/1 "was blank" column for columns with blanks in fit.
    '''

    def __init__(self, strategy="div", divisor=np.sqrt(2), lloq_scale=1.0,
                 add_indicator=False):
        self.strategy = strategy
        self.divisor = divisor
        self.lloq_scale = lloq_scale
        self.add_indicator = add_indicator

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.log_lloq_ = np.nanmin(X, axis=0) + np.log(self.lloq_scale)
        self.blank_cols_ = np.where(np.isnan(X).any(axis=0))[0]
        self.fill_ = np.full(X.shape[1], np.nan)

        for j in range(X.shape[1]):
            c = self.log_lloq_[j]
            if self.strategy == "div":
                self.fill_[j] = c - np.log(self.divisor)
            elif self.strategy == "tobit":
                self.fill_[j] = self._tobit_fill(X[:, j], c)
            else:
                raise ValueError(f"unknown strategy {self.strategy!r}")
        return self

    @staticmethod
    def _tobit_fill(x, c):
        obs = x[~np.isnan(x)]
        n_cens = int(np.isnan(x).sum())
        if n_cens == 0:
            return c

        def nll(p):
            mu, sd = p[0], np.exp(p[1])
            return -(norm.logpdf(obs, mu, sd).sum()
                     + n_cens * norm.logcdf((c - mu) / sd))

        res = minimize(nll, [obs.mean(), np.log(obs.std())], method="Nelder-Mead")
        mu, sd = res.x[0], np.exp(res.x[1])
        a = (c - mu) / sd
        return min(mu - sd * np.exp(norm.logpdf(a) - norm.logcdf(a)), c)

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        mask = np.isnan(X)
        out = np.where(mask, self.fill_, X)
        if self.add_indicator:
            out = np.hstack([out, mask[:, self.blank_cols_].astype(float)])
        return out


def load_prot_features(df_prot: pd.DataFrame) -> pd.DataFrame:
    '''Marker columns only (no id/plate/well), indexed by sample_id.'''

    return df_prot.set_index("sample_id").drop(
        columns=[c for c in PROT_EXCLUDE if c != "sample_id"])


def make_log_transformer() -> FunctionTransformer:
    '''Natural log; NaN-preserving, so it can precede the imputer.'''

    return FunctionTransformer(np.log, feature_names_out="one-to-one")


def make_prot_imputer(strategy: str = "div_sqrt2"):
    '''
    Imputer for log-scale protein features; first step after the log
    transform in the classifier Pipeline so it is fitted on each training
    fold only. "_ind" suffix adds blank-indicator columns.
    '''

    ind = strategy.endswith("_ind")
    name = strategy.removesuffix("_ind")

    if name == "div_sqrt2":
        return LloqImputer("div", np.sqrt(2), add_indicator=ind)
    if name == "div_2":
        return LloqImputer("div", 2.0, add_indicator=ind)
    if name == "lloq":
        return LloqImputer("div", 1.0, add_indicator=ind)
    if name == "tobit":
        return LloqImputer("tobit", add_indicator=ind)
    if name == "median":
        return SimpleImputer(strategy="median", add_indicator=ind)
    if name == "knn":
        return KNNImputer(n_neighbors=5)
    raise ValueError(f"unknown strategy {strategy!r}")


def make_prot_preprocessor(imputer="div_sqrt2") -> Pipeline:
    '''
    log -> impute -> standardize. imputer is a strategy name (see
    make_prot_imputer) or an imputer instance. Standardizing comes after
    imputation so the LLOQ fill logic stays on the log scale.
    '''

    if isinstance(imputer, str):
        imputer = make_prot_imputer(imputer)

    return Pipeline([("log", make_log_transformer()),
                     ("imp", imputer),
                     ("scale", StandardScaler())])


STRATEGIES = ["lloq", "div_2", "div_sqrt2", "div_sqrt2_ind", "tobit", "tobit_ind",
              "median", "knn"]


def evaluate_imputers(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    '''Repeated stratified CV of log -> impute -> scale -> L2 logistic regression.'''

    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=0)
    scoring = {"auc": "roc_auc", "bacc": "balanced_accuracy"}
    rows = {}

    def run(label, pipe):
        s = cross_validate(pipe, X, y, cv=cv, scoring=scoring)
        rows[label] = {
            "auc": f"{s['test_auc'].mean():.3f} +/- {s['test_auc'].std():.3f}",
            "bacc": f"{s['test_bacc'].mean():.3f} +/- {s['test_bacc'].std():.3f}",
        }

    for s in STRATEGIES:
        run(s, Pipeline([("prep", make_prot_preprocessor(s)),
                         ("clf", LogisticRegression(max_iter=1000))]))

    # LLOQ proxy sensitivity: true limit is unknown, so shrink the column min
    for k in (0.8, 0.9):
        run(f"div_sqrt2 (lloq x{k})", Pipeline([
            ("prep", make_prot_preprocessor(
                LloqImputer("div", np.sqrt(2), lloq_scale=k))),
            ("clf", LogisticRegression(max_iter=1000))]))

    # tree model handles NaN natively: reference for "no imputation"
    run("hgb_native_nan", Pipeline([
        ("log", make_log_transformer()),
        ("clf", HistGradientBoostingClassifier(random_state=0))]))

    return pd.DataFrame(rows).T


def censoring_fidelity(X: pd.DataFrame) -> pd.DataFrame:
    '''
    Ground-truth check: censor the lowest q of observed values per marker
    (a higher artificial LLOQ), impute, and report RMSE vs the true log
    values on the censored cells. Imputer only (no standardization), so the
    error stays on the log scale.
    '''

    L = np.log(X.to_numpy(dtype=float))
    rows = {}

    for q in (0.10, 0.25):
        thr = np.nanquantile(L, q, axis=0)
        cens = (L < thr) & ~np.isnan(L)
        Lc = np.where(cens, np.nan, L)

        for s in STRATEGIES:
            imputed = make_prot_imputer(s).fit_transform(Lc)[:, :L.shape[1]]
            rmse = np.sqrt(np.mean((imputed[cens] - L[cens]) ** 2))
            rows.setdefault(s, {})[f"rmse q={q:.2f}"] = round(float(rmse), 3)

    return pd.DataFrame(rows).T


def check_imputer(X: pd.DataFrame) -> None:
    '''Sanity checks: no NaN left, imputed values below LLOQ, fit uses train only.'''

    L = np.log(X.to_numpy(dtype=float))
    for s in ("div_sqrt2", "tobit"):
        imp = make_prot_imputer(s).fit(L[:150])
        out = imp.transform(L)
        assert not np.isnan(out).any(), s
        filled = np.isnan(L)
        assert (out[filled] < np.broadcast_to(imp.log_lloq_, L.shape)[filled]).all(), s
        # held-out rows must not change the fitted limits
        assert np.allclose(imp.log_lloq_, np.nanmin(L[:150], axis=0)), s

    Z = make_prot_preprocessor("div_sqrt2").fit_transform(X.iloc[:150])
    assert np.allclose(Z.mean(axis=0), 0, atol=1e-8)
    assert np.allclose(Z.std(axis=0), 1)


def prot_imputer():

    pwd = Path(__file__).parent

    df_prot = pd.read_csv(Path(pwd, "data/protein.csv"))
    df_train = pd.read_csv(Path(pwd, "data/train_samples.csv"))

    X = load_prot_features(df_prot)
    assert (X.min().min() > 0), "log transform needs positive values"

    train_ids = df_train["sample_id"]
    X = X.loc[train_ids]
    y = df_train.set_index("sample_id").loc[X.index, "cancer_status"].eq("cancer").astype(int)

    check_imputer(X)

    blank = X.isna().mean()
    print(f"{X.shape[0]} train samples, {X.shape[1]} markers; "
          f"blank fraction: {blank.mean():.3f} mean, "
          f"{blank.max():.3f} max ({', '.join(blank[blank > 0.1].index)})\n")

    print("CV (ROC-AUC / balanced accuracy, 5x5 stratified):")
    print(evaluate_imputers(X, y).to_string(), "\n")

    print("Simulated-censoring fidelity (log-scale RMSE, lower is better):")
    print(censoring_fidelity(X).to_string())

    X.reset_index().to_csv(Path(pwd, "data/train_protein.csv"), index=False)


def main() -> int:

    prot_imputer()

    return 0


if __name__ == '__main__':
    sys.exit(main())
