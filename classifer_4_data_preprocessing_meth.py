#!/usr/bin/env python3

'''
##

'''

import sys
import pandas as pd
from pathlib import Path
from sklearn.impute import SimpleImputer


def make_meth_imputer() -> SimpleImputer:
    '''
    Per-column median imputer for methylation features (~2% MCAR missing).
    Use as the first step of the classifier Pipeline so medians are fitted
    on each training fold only.
    '''

    return SimpleImputer(strategy="median")


def data_filtering_meth():

    pwd = Path(__file__).parent

    df_meth = pd.read_csv(Path(pwd, "data/methylation.csv"))
    df_train = pd.read_csv(Path(pwd, "data/train_samples.csv"))

    df_meth = df_meth[df_meth["sample_id"].isin(df_train["sample_id"])]

    df_meth.to_csv(Path(pwd, "data/train_methylation.csv"), index=False)


def main() -> int:

    data_filtering_meth()

    return 0


if __name__ == '__main__':
    sys.exit(main())



