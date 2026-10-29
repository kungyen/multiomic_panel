#!/usr/bin/env python3

'''
##

'''

import sys
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from pathlib import Path


def cohort_visualization():

    pwd = Path(__file__).parent

    sample_df = pd.read_csv(Path(pwd, "data/samples_protein.csv"))

    # one row per patient (age, sex, cancer_status are constant within a patient)
    sample_df = sample_df.drop_duplicates(subset="patient_id")

    # cancer vs non_cancer patients by age and sex
    statuses = ["cancer", "non_cancer"]
    sexes = ["F", "M"]
    colors = {"F": "tab:red", "M": "tab:blue"}
    bins = range(int(sample_df["age"].min() // 5 * 5), int(sample_df["age"].max() // 5 * 5) + 10, 5)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    # top row: age histogram per cancer_status, split by sex
    for ax, status in zip(axes[0], statuses):
        df_status = sample_df[sample_df["cancer_status"] == status]
        ax.hist(
            [df_status.loc[df_status["sex"] == sex, "age"] for sex in sexes],
            bins=bins, stacked=True, color=[colors[sex] for sex in sexes],
            label=sexes, edgecolor="white",
        )
        ax.set_title(f"{status} (n={len(df_status)})")
        ax.set_xlabel("age")
        ax.set_ylabel("patients")
        ax.legend(title="sex")

    # bottom left: sample counts by cancer_status and sex
    counts = sample_df.groupby(["cancer_status", "sex"]).size().unstack("sex").reindex(statuses)[sexes]
    counts.plot.bar(ax=axes[1][0], color=[colors[sex] for sex in sexes], rot=0)
    axes[1][0].set_title("patient count by sex")
    axes[1][0].set_xlabel("")
    axes[1][0].set_ylabel("patients")

    # bottom right: age distribution by cancer_status and sex
    box_data, box_labels, box_colors = [], [], []
    for status in statuses:
        for sex in sexes:
            mask = (sample_df["cancer_status"] == status) & (sample_df["sex"] == sex)
            box_data.append(sample_df.loc[mask, "age"])
            box_labels.append(f"{status}\n{sex}")
            box_colors.append(colors[sex])
    bp = axes[1][1].boxplot(box_data, tick_labels=box_labels, patch_artist=True)
    for patch, color in zip(bp["boxes"], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    axes[1][1].set_title("age by cancer_status and sex")
    axes[1][1].set_ylabel("age")

    fig.suptitle("Cohort: cancer vs non_cancer by age and sex")
    fig.tight_layout()
    plt.savefig(Path(pwd, "data/cohort_visualization.png"), dpi=300)
    plt.show()


def cohort_split():

    pwd = Path(__file__).parent
    sample_df = pd.read_csv(Path(pwd, "data/samples_protein.csv"))

    # split at the patient level so all draws from a patient stay in one partition
    patient_df = sample_df.drop_duplicates(subset="patient_id").copy()

    # stratify on cancer_status x sex x site x age tertile
    # (quartiles leave strata with a single patient, which can't be split)
    patient_df["age_bin"] = pd.qcut(patient_df["age"], q=3, labels=False)
    strata = (
        patient_df["cancer_status"].astype(str)
        + "_" + patient_df["sex"].astype(str)
        + "_" + patient_df["site"].astype(str)
        + "_" + patient_df["age_bin"].astype(str)
    )

    train_ids, test_ids = train_test_split(
        patient_df["patient_id"], test_size=0.2, stratify=strata, random_state=42,
    )

    train_df = sample_df[sample_df["patient_id"].isin(train_ids)].reset_index(drop=True)
    test_df = sample_df[sample_df["patient_id"].isin(test_ids)].reset_index(drop=True)

    # no patient may appear in both partitions
    assert set(train_df["patient_id"]).isdisjoint(test_df["patient_id"])

    print(f"train: {len(train_df)} samples, {train_df['patient_id'].nunique()} patients")
    print(f"test: {len(test_df)} samples, {test_df['patient_id'].nunique()} patients")
    for name, df in [("train", train_df), ("test", test_df)]:
        df_pat = df.drop_duplicates(subset="patient_id")
        print(f"{name} cancer_status:\n{df_pat['cancer_status'].value_counts(normalize=True).round(3)}")
        print(f"{name} sex:\n{df_pat['sex'].value_counts(normalize=True).round(3)}")
        print(f"{name} mean age: {df_pat['age'].mean():.1f}")

    train_df.to_csv(Path(pwd, "data/train_samples.csv"), index=False)
    test_df.to_csv(Path(pwd, "data/test_samples.csv"), index=False)

    return train_df, test_df


def cohort_split_visualization(train_df, test_df):

    pwd = Path(__file__).parent

    # patient level, one row per patient
    df_pat = {
        "train": train_df.drop_duplicates(subset="patient_id").copy(),
        "test": test_df.drop_duplicates(subset="patient_id").copy(),
    }

    # age tertiles from all patients, the same bins cohort_split() stratifies on
    age_all = pd.concat([df["age"] for df in df_pat.values()])
    age_bins = pd.qcut(age_all, q=3)
    upper = [int(iv.right) for iv in age_bins.cat.categories]    # integer ages, bins are (lo, hi]
    lower = [int(age_all.min())] + [hi + 1 for hi in upper[:-1]]
    labels = [f"T{i + 1} ({lo}-{hi})" for i, (lo, hi) in enumerate(zip(lower, upper))]
    edges = [-float("inf")] + upper[:-1] + [float("inf")]
    for df in df_pat.values():
        df["age_group"] = pd.cut(df["age"], bins=edges, labels=labels)
        df["stage"] = df["stage"].fillna("none")    # non_cancer patients have no stage

    features = {
        "age_group": labels,
        "sex": ["F", "M"],
        "site": ["A", "B"],
        "stage": ["none", "I", "II", "III", "IV"],
        "cancer_status": ["cancer", "non_cancer"],
    }
    colors = {"train": "tab:blue", "test": "tab:orange"}
    width = 0.38

    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    axes.flat[-1].axis("off")

    for ax, (feature, categories) in zip(axes.flat, features.items()):
        for i, name in enumerate(["train", "test"]):
            pct = (
                df_pat[name][feature].value_counts(normalize=True)
                .reindex(categories, fill_value=0) * 100
            )
            x = [j + (i - 0.5) * width for j in range(len(categories))]
            bars = ax.bar(x, pct.values, width, color=colors[name],
                          label=f"{name} (n={len(df_pat[name])})")
            ax.bar_label(bars, fmt="%.1f", fontsize=7)
        ax.set_xticks(range(len(categories)))
        ax.set_xticklabels(categories)
        ax.set_title(feature)
        ax.set_ylabel("% of patients")
        ax.set_ylim(0, ax.get_ylim()[1] * 1.2)    # headroom so the legend clears the bars
        ax.legend(loc="upper right", ncol=2)

    fig.suptitle("Train vs test: patient-level composition")
    fig.tight_layout()
    plt.savefig(Path(pwd, "data/cohort_split_visualization.png"), dpi=300)
    plt.show()


def main() -> int:

    # cohort_visualization()

    train_df, test_df = cohort_split()

    cohort_split_visualization(train_df, test_df)

    return 0


if __name__ == '__main__':
    sys.exit(main())



