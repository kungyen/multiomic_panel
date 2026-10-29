#!/usr/bin/env python3

'''
## Sample Filtering

'''

import sys
import pandas as pd
from pathlib import Path

def prot_patients():

    pwd = Path(__file__).parent

    df_sample = pd.read_csv(Path(pwd, "data/samples.csv"))
    df_prot = pd.read_csv(Path(pwd, "data/protein.csv"))

    # Keep only samples whose sample_id appears in Protein.csv
    filtered = df_sample[df_sample["sample_id"].isin(df_prot["sample_id"])]

    # Save and preview
    filtered.to_csv(Path(pwd, "data/samples_protein.csv"), index=False)
    print(filtered.head())


def verify_prot_patients():

    pwd = Path(__file__).parent

    sample_df = pd.read_csv(Path(pwd, "data/samples_protein.csv"))
    check_cols = ["site", "age", "sex", "stage", "cancer_status"]

    # Only patients with more than one row
    multi = sample_df[sample_df.duplicated("patient_id", keep=False)]

    # Number of distinct values per patient for each column (NaN counted as a value)
    n_unique = multi.groupby("patient_id")[check_cols].nunique(dropna=False)

    # Patients where any column has more than one distinct value
    inconsistent_ids = n_unique[(n_unique > 1).any(axis=1)].index

    print(f"{multi['patient_id'].nunique()} patients have multiple rows")
    print(f"{len(inconsistent_ids)} of them have inconsistent values")

    # Show which columns differ for each inconsistent patient
    for pid in inconsistent_ids:
        bad_cols = n_unique.columns[n_unique.loc[pid] > 1].tolist()
        print(f"\npatient_id {pid}: differing columns -> {bad_cols}")
        print(multi.loc[multi["patient_id"] == pid,
                        ["sample_id", "draw_date"] + check_cols].to_string(index=False))


def main() -> int:

    prot_patients()

    # verify_prot_patients()

    return 0


if __name__ == '__main__':
    sys.exit(main())

