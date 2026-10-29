#!/usr/bin/env python3

'''
## Panel selection: normal vs cancer.

Final panel of up to MAX_METH methylation and MAX_PROT protein features,
chosen from the stability-selection candidates that appear in the top lists of
all N_SPLITS outer folds. Greedy forward selection and exhaustive search are
both scored by patient-grouped, repeated CV (L2 logistic regression, AUC), then
compared. The one-standard-error rule picks the smallest panel whose mean AUC
is within one SE of the best panel.

Imputation, log transform and scaling are fitted on each training split only.
They act column by column, so they are fitted once per split on all
candidates and a panel is a column subset of the transformed matrices.

'''

import json
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
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from classifer_4_data_preprocessing_meth import make_meth_imputer
from classifer_4_data_preprocessing_prot import make_log_transformer, make_prot_imputer
from classifer_5_logireg_meth_only import N_SPLITS, SEED

MAX_METH = 5
MAX_PROT = 3
N_REPEATS = 5       # repeated grouped CV; N_REPEATS * N_SPLITS scored splits
C = 1.0


class PanelScorer:
    '''
    Cross-validated scores of feature subsets, cached by subset.
    '''

    def __init__(self, df: pd.DataFrame, meth_cols, prot_cols):

        self.cols = list(meth_cols) + list(prot_cols)
        self.is_meth = {c: c in set(meth_cols) for c in self.cols}
        self.y = (df["cancer_status"] == "cancer").astype(int).to_numpy()
        self.splits = []
        self.cache = {}

        groups = df["patient_id"].to_numpy()
        X = df[self.cols]

        for r in range(N_REPEATS):
            cv = StratifiedGroupKFold(N_SPLITS, shuffle=True, random_state=SEED + r)
            for tr, te in cv.split(X, self.y, groups):
                assert not set(groups[tr]) & set(groups[te])
                prep = Pipeline([
                    ("prep", ColumnTransformer([
                        ("meth", make_meth_imputer(), list(meth_cols)),
                        ("prot", Pipeline([("log", make_log_transformer()),
                                           ("imp", make_prot_imputer("div_sqrt2"))]),
                         list(prot_cols)),
                    ])),
                    ("scale", StandardScaler()),
                ]).fit(X.iloc[tr])
                self.splits.append((prep.transform(X.iloc[tr]), self.y[tr],
                                    prep.transform(X.iloc[te]), self.y[te]))

    def _score(self, idx: tuple) -> dict:
        '''
        Per-split scores of the panel given by column indices.
        '''

        idx = list(idx)
        auc, bacc, ll = [], [], []

        for Xtr, ytr, Xte, yte in self.splits:
            clf = LogisticRegression(C=C, class_weight="balanced", max_iter=1000)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                clf.fit(Xtr[:, idx], ytr)
            p = clf.predict_proba(Xte[:, idx])[:, 1]
            auc.append(roc_auc_score(yte, p))
            bacc.append(balanced_accuracy_score(yte, p > 0.5))
            ll.append(log_loss(yte, p, labels=[0, 1]))

        auc = np.array(auc)
        return {
            "auc": auc.mean(),
            # repeats reuse the same patients, so the effective n is the fold count
            "auc_se": auc.std(ddof=1) / np.sqrt(N_SPLITS),
            "bacc": np.mean(bacc),
            "log_loss": np.mean(ll),
        }

    def score_many(self, panels: list[frozenset]) -> None:

        todo = [p for p in panels if p not in self.cache]
        pos = {c: i for i, c in enumerate(self.cols)}
        res = Parallel(n_jobs=-1)(
            delayed(self._score)(tuple(sorted(pos[c] for c in p))) for p in todo)
        self.cache.update(zip(todo, res))

    def table(self, panels: list[frozenset]) -> pd.DataFrame:

        self.score_many(panels)
        rows = []
        for p in panels:
            rows.append({
                "n_meth": sum(self.is_meth[c] for c in p),
                "n_prot": sum(not self.is_meth[c] for c in p),
                "size": len(p),
                "features": ";".join(sorted(p, key=self.cols.index)),
                **self.cache[p],
            })
        return pd.DataFrame(rows)


def fold_stable_candidates(path: Path, n_folds: int) -> list[str]:
    '''
    Features that are in the top list of every outer fold, in the order of
    their best rank.
    '''

    cands = pd.read_csv(path)
    n = cands.groupby("feature")["fold"].nunique()
    best_rank = cands.groupby("feature")["rank"].mean()

    return best_rank[n[n == n_folds].index].sort_values().index.tolist()


def one_se_pick(tab: pd.DataFrame) -> pd.Series:
    '''
    Smallest panel whose mean AUC is within one SE of the best panel; ties on
    size go to the higher mean.
    '''

    best = tab.loc[tab["auc"].idxmax()]
    ok = tab[tab["auc"] >= best["auc"] - best["auc_se"]]

    return ok.sort_values(["size", "auc"], ascending=[True, False]).iloc[0]


def greedy_selection(scorer: PanelScorer, meth_cols, prot_cols) -> pd.DataFrame:
    '''
    Forward selection: at each step add the candidate with the best mean AUC,
    respecting the per-modality caps, until both caps are reached.
    '''

    chosen, rows = frozenset(), []

    while True:
        n_m = sum(c in meth_cols for c in chosen)
        n_p = len(chosen) - n_m
        pool = ([c for c in meth_cols if c not in chosen] if n_m < MAX_METH else []) \
             + ([c for c in prot_cols if c not in chosen] if n_p < MAX_PROT else [])
        if not pool:
            break

        trials = [chosen | {c} for c in pool]
        scorer.score_many(trials)
        best = max(zip(pool, trials), key=lambda t: scorer.cache[t[1]]["auc"])
        chosen = best[1]

        rows.append({"step": len(chosen), "added": best[0],
                     "features": ";".join(sorted(chosen, key=scorer.cols.index)),
                     "n_meth": n_m + (best[0] in meth_cols),
                     "n_prot": n_p + (best[0] in prot_cols),
                     "size": len(chosen), **scorer.cache[chosen]})

    return pd.DataFrame(rows)


def exhaustive_selection(scorer: PanelScorer, meth_cols, prot_cols) -> pd.DataFrame:
    '''
    Every non-empty panel with at most MAX_METH methylation and MAX_PROT
    protein features.
    '''

    panels = []
    for m in range(min(MAX_METH, len(meth_cols)) + 1):
        for p in range(min(MAX_PROT, len(prot_cols)) + 1):
            if m + p:
                panels += [frozenset(a + b) for a in combinations(meth_cols, m)
                           for b in combinations(prot_cols, p)]

    return scorer.table(panels)


def compare(greedy: pd.DataFrame, exh: pd.DataFrame) -> pd.DataFrame:
    '''
    Per panel size: greedy step vs best exhaustive panel of that size.
    '''

    best = exh.loc[exh.groupby("size")["auc"].idxmax()].set_index("size")
    g = greedy.set_index("size")
    out = pd.DataFrame({
        "greedy_features": g["features"], "greedy_auc": g["auc"],
        "exhaustive_features": best["features"], "exhaustive_auc": best["auc"],
    })
    out["gap"] = out["exhaustive_auc"] - out["greedy_auc"]
    out["same_panel"] = [set(a.split(";")) == set(b.split(";"))
                         for a, b in zip(out["greedy_features"], out["exhaustive_features"])]

    return out.reset_index()


def plot_selection(cmp: pd.DataFrame, exh: pd.DataFrame, greedy: pd.DataFrame,
                   picks: dict, out_path: Path) -> None:

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.scatter(exh["size"], exh["auc"], s=6, color="0.8", label="all panels")
    ax.plot(cmp["size"], cmp["exhaustive_auc"], marker="o", label="exhaustive best")
    ax.errorbar(greedy["size"] + 0.08, greedy["auc"], yerr=greedy["auc_se"],
                marker="s", ls="--", capsize=2, label="greedy path")
    for name, color in [("greedy", "tab:green"), ("exhaustive", "tab:red")]:
        ax.axvline(picks[name]["size"], color=color, ls=":", lw=1,
                   label=f"{name} 1-SE pick")
    best = exh.loc[exh["auc"].idxmax()]
    ax.axhspan(best["auc"] - best["auc_se"], best["auc"], color="tab:blue", alpha=0.1)
    ax.set(xlabel="panel size", ylabel="CV AUC",
           title="Panel selection (shaded: 1 SE of best panel)")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def panel_workflow(df_samples, df_meth, df_prot, out_dir: Path):

    meth_cols = fold_stable_candidates(Path(out_dir, "meth_stability_fold_candidates.csv"), N_SPLITS)
    prot_cols = fold_stable_candidates(Path(out_dir, "prot_stability_fold_candidates.csv"), N_SPLITS)
    print(f"candidates in all {N_SPLITS} folds: {len(meth_cols)} methylation, "
          f"{len(prot_cols)} protein")
    print("  meth:", ", ".join(meth_cols))
    print("  prot:", ", ".join(prot_cols))

    df = (df_samples.merge(df_meth[["sample_id"] + meth_cols], on="sample_id", how="inner")
                    .merge(df_prot[["sample_id"] + prot_cols], on="sample_id", how="inner"))
    scorer = PanelScorer(df, meth_cols, prot_cols)

    greedy = greedy_selection(scorer, meth_cols, prot_cols)
    exh = exhaustive_selection(scorer, meth_cols, prot_cols)
    print(f"\nexhaustive: {len(exh)} panels scored")

    cmp = compare(greedy, exh)
    print("\ngreedy vs exhaustive best per size:")
    print(cmp.drop(columns=["greedy_features", "exhaustive_features"])
             .to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    picks = {"greedy": one_se_pick(greedy), "exhaustive": one_se_pick(exh)}
    for name, pick in picks.items():
        best = (greedy if name == "greedy" else exh)["auc"].max()
        print(f"\n{name} 1-SE pick: size {pick['size']} "
              f"({pick['n_meth']} meth + {pick['n_prot']} prot)  "
              f"AUC {pick['auc']:.3f} (best {best:.3f}, SE {pick['auc_se']:.3f})\n"
              f"  {pick['features']}")

    sets = {k: set(v["features"].split(";")) for k, v in picks.items()}
    jac = len(sets["greedy"] & sets["exhaustive"]) / len(sets["greedy"] | sets["exhaustive"])
    print(f"\npick Jaccard greedy vs exhaustive: {jac:.2f}")

    paths = {
        "greedy_path": Path(out_dir, "panel_greedy_path.csv"),
        "exhaustive_all": Path(out_dir, "panel_exhaustive_all.csv"),
        "comparison": Path(out_dir, "panel_comparison.csv"),
        "final_csv": Path(out_dir, "panel_final.csv"),
        "final_json": Path(out_dir, "panel_final.json"),
        "plot": Path(out_dir, "panel_selection.png"),
    }
    greedy.to_csv(paths["greedy_path"], index=False)
    exh.sort_values("auc", ascending=False).to_csv(paths["exhaustive_all"], index=False)
    cmp.to_csv(paths["comparison"], index=False)

    final = [{"method": name, "modality": "methylation" if f in meth_cols else "protein",
              "feature": f, "panel_auc": pick["auc"], "panel_auc_se": pick["auc_se"]}
             for name, pick in picks.items() for f in pick["features"].split(";")]
    pd.DataFrame(final).to_csv(paths["final_csv"], index=False)

    lists = {name: {"methylation": [f for f in sorted(s) if f in meth_cols],
                    "protein": [f for f in sorted(s) if f in prot_cols]}
             for name, s in sets.items()}
    paths["final_json"].write_text(json.dumps(lists, indent=2))

    plot_selection(cmp, exh, greedy, picks, paths["plot"])

    print("\nsaved:")
    for k, v in paths.items():
        print(f"  {k}: {v}")
    print("\nfinal panel (exhaustive 1-SE):")
    print(json.dumps(lists["exhaustive"], indent=2))

    return lists


def main() -> int:

    pwd = Path(__file__).parent

    df_samples = pd.read_csv(Path(pwd, "data/train_samples.csv"))

    df_meth = pd.read_csv(Path(pwd, "data/train_methylation.csv"))

    df_prot = pd.read_csv(Path(pwd, "data/train_protein.csv"))

    panel_workflow(df_samples, df_meth, df_prot, Path(pwd, "data"))

    return 0


if __name__ == '__main__':
    sys.exit(main())
