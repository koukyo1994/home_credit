import gc
import os
import time
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgbm
from scipy.io import savemat, loadmat
from scipy.sparse import csr_matrix, csc_matrix, hstack
from contextlib import contextmanager
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold
warnings.simplefilter(action="ignore", category=FutureWarning)


@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print(f"{title} - done in {time.time()-t0:.2f}")


def one_hot_encoder(df, nan_as_category=True, sparse=True):
    columns = df.columns
    categoricals = [col for col in columns if df[col].dtype="object"]
    new_df = pd.get_dummies(df,
                            columns=categoricals,
                            dummy_na=nan_as_category)
    new_columns = [col for col in new_df.columns]
    return new_df, new_columns


def car_owner_w_gender(df):
    f = lambda x, sex: 1 if (x["CODE_GENDER"] == sex and
                             x["FLAG_OWN_CAR"] == 'Y') else 0
    car_owner_m = df.apply(lambda x: f(x, 'M'), axis=1)
    car_owner_f = df.apply(lambda x: f(x, 'F'), axis=1)
    new_df = pd.DataFrame({
        "car_owner_m": car_owner_m,
        "car_owner_f": car_owner_f
    })
    return new_df


def realty_own_w_gender(df):
    f = lambda x, sex: 1 if (x["CODE_GENDER"] == sex and
                             x["FLAG_OWN_REALTY"] == 'Y') else 0
    realty_own_m = df.apply(lambda x: f(x, 'M'), axis=1)
    realty_own_f = df.apply(lambda x: f(x, 'F'), axis=1)
    new_df = pd.DataFrame({
        "realty_own_m": realty_own_m,
        "realty_own_f": realty_own_f
    })
    return new_df


def has_children_w_gender(df):
    f = lambda x, sex: 1 if (x["CODE_GENDER"] == sex and
                             x["CNT_CHILDREN"] > 0) else 0
    has_child_m = df.apply(lambda x: f(x, 'M'), axis=1)
    has_child_f = df.apply(lambda x: f(x, 'F'), axis=1)
    new_df = pd.DataFrame({
        "has_child_m": has_child_m,
        "has_child_f": has_child_f
    })
    return new_df


def has_many_children_w_gender(df, n_child_thres):
    f = lambda x, sex: 1 if (x["CODE_GENDER"] == sex and
                             x["CNT_CHILDREN"] > n_child_thres) else 0
    has_many_children_m = df.apply(lambda x: f(x, 'M'), axis=1)
    has_many_children_f = df.apply(lambda x: f(x, 'F'), axis=1)
    new_df = pd.DataFrame({
        "has_many_children_m": has_many_children_m,
        "has_many_children_f": has_many_children_f
    })
    return new_df
