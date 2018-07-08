import numpy as np
import pandas as pd


def check_has_null(df):
    cols = df.columns
    has_null = dict()
    for col in cols:
        null_mean = df[col].isnull().mean()
        if null_mean > 0.0:
            has_null[col] = null_mean
    for key, val in has_null.items():
        print(f"{key}: {val}")
