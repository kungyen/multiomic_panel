from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedKFold, cross_val_predict
from statsmodels.stats.multitest import multipletests

pwd = Path(__file__).parent


def load():
    prot = pd.read_csv(Path(pwd, "data/protein.csv"))
    meta = pd.read_csv(Path(pwd, "data/samples_protein.csv"))
    df = prot.merge(meta, on="sample_id", how="left", validate="1:1")
    df["row"] = df["well"].str[0]
    df["col"] = df["well"].str[1:].astype(int)
    analytes = list(prot.columns[3:])
    return df, analytes


def kruskal_by(logx, df, analytes, group):
    p = np.array([stats.kruskal(*[x.dropna() for _, x in logx[a].groupby(df[group])]).pvalue
                  for a in analytes])
    q = multipletests(p, method="fdr_bh")[1]
    top = [analytes[i] for i in np.argsort(p)[:4]]
    print(f"{group}: {(p < 0.05).sum()}/{len(p)} analytes p<0.05, {(q < 0.05).sum()} BH<0.05, "
          f"min p={p.min():.1e}, top {top}")


def pc_associations(pcs, df):
    for i in range(pcs.shape[1]):
        plate = stats.kruskal(*[pcs[(df["plate_id"] == k).values, i] for k in df["plate_id"].unique()]).pvalue
        row = stats.kruskal(*[pcs[(df["row"] == k).values, i] for k in df["row"].unique()]).pvalue
        rho, col = stats.spearmanr(pcs[:, i], df["col"])
        print(f"PC{i + 1}: plate p={plate:.1e}, row p={row:.2f}, column rho={rho:.2f} p={col:.2f}")


def confounding(df):
    for col in ["site", "cancer_status"]:
        tab = pd.crosstab(df["plate_id"], df[col])
        print(f"\nplate x {col}\n{tab}\nchi2 p={stats.chi2_contingency(tab)[1]:.1e}")


def plate_vs_site(logx, df, analytes):
    # PLT02 is the only plate with both sites, so site differences there are free of plate
    sub = df["plate_id"] == "PLT02"
    a, b = (sub & (df["site"] == s) for s in ["A", "B"])
    p = np.array([stats.mannwhitneyu(logx.loc[a, x].dropna(), logx.loc[b, x].dropna()).pvalue
                  for x in analytes])
    print(f"\nwithin PLT02, site A (n={a.sum()}) vs B (n={b.sum()}): "
          f"{(p < 0.05).sum()}/{len(p)} analytes p<0.05, "
          f"{(multipletests(p, method='fdr_bh')[1] < 0.05).sum()} BH<0.05")


def adjusted_cancer_effect(logx, df, analytes):
    d = logx.copy()
    d["plate_id"], d["cancer"] = df["plate_id"], (df["cancer_status"] == "cancer").astype(int)
    res = {}
    for a in analytes:
        fit = smf.ols(f"{a} ~ C(plate_id) + cancer", data=d.dropna(subset=[a])).fit()
        res[a] = fit.pvalues["cancer"]
    p = pd.Series(res)
    n_adj = (multipletests(p, method="fdr_bh")[1] < 0.05).sum()
    p_raw = pd.Series({a: stats.mannwhitneyu(d.loc[d["cancer"] == 1, a].dropna(),
                                             d.loc[d["cancer"] == 0, a].dropna()).pvalue
                       for a in analytes})
    n_raw = (multipletests(p_raw, method="fdr_bh")[1] < 0.05).sum()
    print(f"\ncancer vs non_cancer, analytes BH<0.05: {n_raw} unadjusted, {n_adj} adjusted for plate")


def plate_center(logx, df):
    return logx - logx.groupby(df["plate_id"]).transform("median")


def cancer_cv_auc(X, df, seed=0):
    y = (df["cancer_status"] == "cancer").astype(int).values
    X = (X - X.mean()) / X.std()
    # patients repeat, so folds are grouped by patient; plate effect is not held out
    oof = np.zeros(len(y))
    for tr, te in GroupKFold(5).split(X, y, df["patient_id"]):
        m = LogisticRegression(C=0.1, max_iter=2000).fit(X.iloc[tr], y[tr])
        oof[te] = m.predict_proba(X.iloc[te])[:, 1]
    return roc_auc_score(y, oof)


def plots(Z, Zc, df):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, data, title in [(axes[0], Z, "PCA, raw (by plate)"), (axes[1], Zc, "PCA, plate-centered (by plate)")]:
        pcs = PCA(2).fit_transform(data)
        for k, g in df.groupby("plate_id"):
            ax.scatter(pcs[(df["plate_id"] == k).values, 0], pcs[(df["plate_id"] == k).values, 1], s=12, label=k)
        ax.set_title(title)
        ax.legend()
    pd.crosstab(df["plate_id"], df["cancer_status"]).plot.bar(ax=axes[2], rot=0, stacked=True)
    axes[2].set_title("samples by plate and cancer_status")
    fig.tight_layout()
    fig.savefig(Path(pwd, "data/batch_check.png"), dpi=150)


def main():
    df, analytes = load()
    print(f"{len(df)} samples, {df['patient_id'].nunique()} patients, plates {df['plate_id'].value_counts().to_dict()}")
    logx = np.log2(df[analytes].clip(lower=1e-3))
    print(f"zero/negative values clipped: {(df[analytes] < 1e-3).sum().sum()}")

    for group in ["plate_id", "row", "col"]:
        kruskal_by(logx, df, analytes, group)

    Z = logx.fillna(logx.median())
    Z = (Z - Z.mean()) / Z.std()
    pca = PCA(5).fit(Z)
    print(f"\nPC variance explained: {pca.explained_variance_ratio_.round(3)}")
    pc_associations(pca.transform(Z), df)

    plate = df["plate_id"].values
    pred = cross_val_predict(LogisticRegression(C=0.1, max_iter=2000), Z, plate,
                             cv=StratifiedKFold(5, shuffle=True, random_state=0))
    print(f"\nplate predicted from proteins: accuracy {(pred == plate).mean():.2f}, "
          f"chance {df['plate_id'].value_counts(normalize=True).max():.2f}")

    confounding(df)
    plate_vs_site(logx, df, analytes)
    adjusted_cancer_effect(logx, df, analytes)

    Lc = plate_center(logx, df)
    Zc = Lc.fillna(0)
    Zc = (Zc - Zc.mean()) / Zc.std()
    print("\nafter plate median-centering:")
    pc_associations(PCA(5).fit_transform(Zc), df)
    plate_dummies = pd.get_dummies(df["plate_id"]).astype(float)
    print(f"\ngrouped-CV AUC for cancer_status: raw {cancer_cv_auc(Z, df):.2f}, "
          f"plate-centered {cancer_cv_auc(Zc, df):.2f}, plate label only {cancer_cv_auc(plate_dummies, df):.2f}")
    plots(Z, Zc, df)


if __name__ == "__main__":
    main()
