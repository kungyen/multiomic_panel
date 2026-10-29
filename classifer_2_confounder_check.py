from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

pwd = Path(__file__).parent


def patient_table(df):
    # site/age/sex/cancer_status should be constant within a patient
    for col in ["site", "age", "sex", "cancer_status"]:
        n_bad = (df.groupby("patient_id")[col].nunique() > 1).sum()
        if n_bad:
            print(f"WARNING: {n_bad} patients have >1 value of {col}")
    return df.groupby("patient_id").first().reset_index()


def categorical_test(pat, col):
    tab = pd.crosstab(pat[col], pat["cancer_status"])
    print(f"\n{col} x cancer_status\n{tab}")
    print(f"row %:\n{(tab.div(tab.sum(axis=1), axis=0) * 100).round(1)}")
    chi2, p, _, _ = stats.chi2_contingency(tab)
    cramers_v = np.sqrt(chi2 / (tab.values.sum() * (min(tab.shape) - 1)))
    print(f"chi2 p={p:.2e}, Cramer's V={cramers_v:.2f}")
    if tab.shape == (2, 2):
        odds, p_f = stats.fisher_exact(tab)
        print(f"Fisher p={p_f:.2e}, odds ratio={odds:.2f}")


def age_test(pat):
    c = pat.loc[pat["cancer_status"] == "cancer", "age"]
    n = pat.loc[pat["cancer_status"] == "non_cancer", "age"]
    _, p = stats.mannwhitneyu(c, n)
    pooled = np.sqrt((c.var() * (len(c) - 1) + n.var() * (len(n) - 1)) / (len(c) + len(n) - 2))
    auc = roc_auc_score(pat["cancer_status"] == "cancer", pat["age"])
    print(f"\nage: cancer {c.mean():.1f} vs non_cancer {n.mean():.1f}; "
          f"Mann-Whitney p={p:.2e}, Cohen's d={(c.mean() - n.mean()) / pooled:.2f}, AUC(age alone)={auc:.2f}")


def covariate_only_cv_auc(pat, seed=0):
    # patient-level folds, so samples from one patient never straddle train/test
    X = pd.get_dummies(pat[["site", "sex"]], drop_first=True).astype(float)
    X["age"] = (pat["age"] - pat["age"].mean()) / pat["age"].std()
    y = (pat["cancer_status"] == "cancer").astype(int).values
    site = [c for c in X if c.startswith("site")]
    sex = [c for c in X if c.startswith("sex")]
    for name, cols in {"site only": site,
                       "age only": ["age"],
                       "sex only": sex,
                       "site+age": site + ["age"],
                       "site+sex": site + sex,
                       "age+sex": ["age"] + sex,
                       "site+age+sex": list(X.columns)}.items():
        oof = np.zeros(len(y))
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y):
            m = LogisticRegression().fit(X.iloc[tr][cols], y[tr])
            oof[te] = m.predict_proba(X.iloc[te][cols])[:, 1]
        print(f"CV AUC, {name}: {roc_auc_score(y, oof):.2f}")


def plots(pat):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    pd.crosstab(pat["site"], pat["cancer_status"]).plot.bar(ax=axes[0], rot=0, stacked=True)
    axes[0].set_title("patients by site and cancer_status")
    pd.crosstab(pat["sex"], pat["cancer_status"]).plot.bar(ax=axes[1], rot=0, stacked=True)
    axes[1].set_title("patients by sex and cancer_status")
    groups = {f"{s}\n{c}": g["age"] for (s, c), g in pat.groupby(["site", "cancer_status"])}
    axes[2].boxplot(groups.values(), tick_labels=groups.keys())
    axes[2].set_title("age by site and cancer_status")
    fig.tight_layout()
    fig.savefig(Path(pwd, "data/confounder_check.png"), dpi=150)


def main():
    df = pd.read_csv(Path(pwd, "data/train_samples.csv"))
    pat = patient_table(df)
    print(f"{len(df)} samples, {len(pat)} patients")

    for col in ["site", "sex"]:
        categorical_test(pat, col)
    age_test(pat)

    pat["cancer"] = (pat["cancer_status"] == "cancer").astype(int)
    fit = smf.logit("cancer ~ C(site) + age + C(sex)", data=pat).fit(disp=0)
    ci = np.exp(fit.conf_int())
    print("\nlogistic regression odds ratios (cancer ~ site + age + sex)")
    print(pd.DataFrame({"OR": np.exp(fit.params), "lo": ci[0], "hi": ci[1], "p": fit.pvalues}).round(3))

    print()
    covariate_only_cv_auc(pat)

    print(f"\nage by site:\n{pat.groupby('site')['age'].describe().round(1)}")
    print(f"\nsex x site:\n{pd.crosstab(pat['site'], pat['sex'])}")
    print(f"\nstage x site:\n{pd.crosstab(pat['site'], pat['stage'].fillna('none'))}")
    plots(pat)


if __name__ == "__main__":
    main()
