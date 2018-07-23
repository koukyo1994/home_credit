import gc
import os
import time
import ipdb
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from lightgbm import LGBMClassifier
from contextlib import contextmanager
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold
warnings.simplefilter(action="ignore", category=FutureWarning)


# Constants or Hyper Parameters
DELAY_IMPORTANCE = 2
TIME_CONSTANT = 12


class TargetEncoder:
    def __init__(self):
        self.encoder = None

    def fit(self, cat, target):
        df = pd.concat([cat, target], axis=1)
        cname = cat.name
        tname = target.name

        self.encoder = df.groupby(cname)[tname].mean()

    def transform(self, cat):
        return cat.map(lambda x: self.encoder[x])

    def fit_transform(self, cat, target):
        self.fit(cat, target)
        return self.transform(cat)


@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print(f"{title} - done in {time.time()-t0:.2f}")


def one_hot_encoder(df, use_cols=None, nan_as_category=True):
    columns = use_cols if use_cols else df.columns
    categoricals = [col for col in columns if df[col].dtype == 'object']
    new_df = pd.get_dummies(df,
                            columns=categoricals,
                            dummy_na=nan_as_category)
    new_columns = [col for col in new_df.columns]
    return new_df, new_columns


def calc_delay_seriousity(df):
    df["STATUS_INT"] = df["STATUS"].map(lambda x: '0' if (x == 'C' or x == 'X')
                                        else x)
    df["STATUS_INT"] = df["STATUS_INT"].apply(lambda x: int(x))

    df["DELAY_SERIOUSITY"] = np.exp(df["MONTHS_BALANCE"]/TIME_CONSTANT)  * \
                             df["STATUS_INT"] ** DELAY_IMPORTANCE
    return df


def bureau_and_balance(num_rows=None, nan_as_category=True):
    # read the csv files
    br = pd.read_csv("../data/bureau.csv", nrows=num_rows)
    bb = pd.read_csv("../data/bureau_balance.csv", nrows=num_rows)
    # Activeな最後の月(ここで抽出しないとone hot vectorizeで消える)
    bb_active_last_month = bb.query('STATUS != "C"').groupby("SK_ID_BUREAU") \
                           ["MONTHS_BALANCE"].max()

    # 過去のdelayの重大度の計算
    bb = calc_delay_seriousity(bb)

    # make the STATUS columns one-hot-vectorized
    bb, bbcat = one_hot_encoder(bb, use_cols=["STATUS"], nan_as_category=False)

    # aggregations dict will be -> MONTHS_BALANCE: size, STATUS_*: sum
    bb_aggregations = dict()
    bbcat.remove("SK_ID_BUREAU")
    bbcat.remove("MONTHS_BALANCE")
    for col in bbcat:
        bb_aggregations[col] = ['sum']
    bb_aggregations["MONTHS_BALANCE"] = ['size']
    bb_aggregations["DELAY_SERIOUSITY"] = ['sum']
    bb_agg = bb.groupby("SK_ID_BUREAU").agg(bb_aggregations)

    # 列名の階層性をなくす処理
    bb_agg.columns = pd.Index(e[0] + "_" + e[1].upper()
                              for e in bb_agg.columns.tolist())

    # STATUSがCになるとその後はずっとC -> Closedの意味なのでその後の部分はいらない
    bb_agg["ACTIVE_MONTHS"] = bb_agg["MONTHS_BALANCE_SIZE"] - \
                              bb_agg["STATUS_C_SUM"]
    bb_agg.drop(["MONTHS_BALANCE_SIZE", "STATUS_C_SUM"],
                axis=1, inplace=True)
    bbcat.remove("STATUS_C")
    for col in bbcat:
        colname = col + "_RATIO"
        colname_sum = col + "_SUM"
        bb_agg[colname] = bb_agg[colname_sum] / bb_agg["ACTIVE_MONTHS"]

    bb_agg["DELAYED_ONCE_OR_MORE"] = bb_agg["STATUS_1_SUM"] + bb_agg["STATUS_2_SUM"] + \
                                     bb_agg["STATUS_2_SUM"] + bb_agg["STATUS_3_SUM"] + \
                                     bb_agg["STATUS_4_SUM"] + bb_agg["STATUS_5_SUM"]
    bb_agg["DLYED_OOM_RATIO"] = bb_agg["DELAYED_ONCE_OR_MORE"] / bb_agg["ACTIVE_MONTHS"]
    bb_agg["WEIGHTED_DELAY_SUM"] = bb_agg["STATUS_1_SUM"] * 1 + \
                                   bb_agg["STATUS_2_SUM"] * 2 + \
                                   bb_agg["STATUS_3_SUM"] * 3 + \
                                   bb_agg["STATUS_4_SUM"] * 4 + \
                                   bb_agg["STATUS_5_SUM"] * 5
    bb_agg["ACTIVE_LAST_MONTH"] = bb_active_last_month
    bb_agg = bb_agg.reset_index()

    br = br.merge(bb_agg, how='left', on='SK_ID_BUREAU')
    del bb, bb_agg
    gc.collect()

    # Feature adding (Stated in the Shanth's kernel)
    false_active = br.query('ACTIVE_LAST_MONTH < 0 & CREDIT_ACTIVE != "Closed"')
    br.loc[false_active.index, "CREDIT_ACTIVE"] = "Closed"

    br["CREDIT_ACTIVE_BINARY"] = br["CREDIT_ACTIVE"].map(
        lambda x: 0 if x == "Closed" else 1
    )
    br["CREDIT_ENDDATE_BINARY"] = br["DAYS_CREDIT_ENDDATE"].map(
        lambda x: 0 if x < 0 else 1
    )
    br["AMT_CREDIT_SUM_DEBT"] = br["AMT_CREDIT_SUM_DEBT"].fillna(0)
    br["AMT_CREDIT_SUM"] = br["AMT_CREDIT_SUM"].fillna(0)
    br["CNT_CREDIT_PROLONG"] = br["CNT_CREDIT_PROLONG"].fillna(0)

    # Feature 1, 2, 4, 6, 8, 10
    aggregations = {
        "DAYS_CREDIT": ['count'],
        "CREDIT_TYPE": ['nunique'],
        "CREDIT_ACTIVE_BINARY": ['mean'],
        "CREDIT_ENDDATE_BINARY": ['mean'],
        "AMT_CREDIT_SUM_DEBT": ['sum'],
        "AMT_CREDIT_SUM": ['sum'],
        "AMT_CREDIT_SUM_OVERDUE": ['sum'],
        "CNT_CREDIT_PROLONG": ['mean']
    }

    br_agg1 = br.groupby("SK_ID_CURR").agg(aggregations)
    br_agg1.columns = pd.Index([e[0] + "_" + e[1].upper() for e in br_agg1.columns.tolist()])
    br_agg1 = br_agg1.reset_index().rename(columns={
        "DAYS_CREDIT_COUNT": "BUREAU_LOAN_COUNT",
        "CREDIT_TYPE_NUNIQUE": "BUREAU_LOAN_TYPE",
        "CREDIT_ACTIVE_BINARY_MEAN": "ACTIVE_LOANS_PERCENTAGE",
        "CREDIT_ENDDATE_BINARY_MEAN": "CREDIT_ENDDATE_PERCENTAGE",
        "AMT_CREDIT_SUM_DEBT_SUM": "TOTAL_CUSTOMER_DEBT",
        "AMT_CREDIT_SUM_SUM": "TOTAL_CUSTOMER_CREDIT",
        "AMT_CREDIT_SUM_OVERDUE_SUM": "TOTAL_CUSTOMER_OVERDUE",
        "CNT_CREDIT_PROLONG_MEAN": "AVG_CREDITDAYS_PROLONGED"
    })
    br = br.merge(br_agg1, how='left', on='SK_ID_CURR')
    del br_agg1
    gc.collect()

    # Feature 3, 8
    br["AVERAGE_LOAN_TYPE"] = br["BUREAU_LOAN_COUNT"]/br["BUREAU_LOAN_TYPE"]
    br["DEBT_CREDIT_RATIO"] = br["TOTAL_CUSTOMER_DEBT"]/br["TOTAL_CUSTOMER_CREDIT"]
    br["OVERDUE_DEBT_RATIO"] = br["TOTAL_CUSTOMER_OVERDUE"]/br["TOTAL_CUSTOMER_DEBT"]
    del br["TOTAL_CUSTOMER_DEBT"], br["TOTAL_CUSTOMER_CREDIT"], br["TOTAL_CUSTOMER_OVERDUE"]
    gc.collect()

    # Feature 5

    grp = br[["SK_ID_CURR", "SK_ID_BUREAU", "DAYS_CREDIT"]].groupby("SK_ID_CURR")
    grp = grp.apply(lambda x: x.sort_values("DAYS_CREDIT", ascending=False)) \
          .reset_index(drop=True)
    grp["DAYS_CREDIT"] *= -1
    grp["DAYS_DIFF"] = grp.groupby("SK_ID_CURR")["DAYS_CREDIT"].diff()
    grp["DAYS_DIFF"] = grp["DAYS_DIFF"].fillna(0).astype('uint32')
    del grp["DAYS_CREDIT"], grp["SK_ID_CURR"]
    gc.collect()
    br = br.merge(grp, how='left', on='SK_ID_BUREAU')

    # Feature 7
    br1 = br.query('CREDIT_ENDDATE_BINARY == 1')
    grp = br1[["SK_ID_BUREAU", "SK_ID_CURR", "DAYS_CREDIT_ENDDATE"]] \
          .groupby("SK_ID_CURR").apply(
              lambda x: x.sort_values("DAYS_CREDIT_ENDDATE", ascending=True)
          ).reset_index(drop=True)
    grp["DAYS_ENDDATE_DIFF"] = grp.groupby("SK_ID_CURR")["DAYS_CREDIT_ENDDATE"].diff()
    grp["DAYS_ENDDATE_DIFF"] = grp["DAYS_ENDDATE_DIFF"].fillna(0).astype('uint32')
    del grp["DAYS_CREDIT_ENDDATE"], grp["SK_ID_CURR"]
    gc.collect()
    br = br.merge(grp, how='left', on='SK_ID_BUREAU')

    num_aggregations = {
        'DAYS_CREDIT': ['mean'],
        'CREDIT_DAY_OVERDUE': ['mean', 'sum', 'max'],
        'DAYS_CREDIT_ENDDATE': ['mean'],
        'DAYS_CREDIT_UPDATE': ['mean', 'max'],
        'AMT_CREDIT_MAX_OVERDUE': ['mean', 'sum'],
        'CNT_CREDIT_PROLONG': ['sum'],
        'AMT_CREDIT_SUM': ['mean', 'sum'],
        'AMT_CREDIT_SUM_DEBT': ['mean', 'sum'],
        'AMT_CREDIT_SUM_OVERDUE': ['mean'],
        'AMT_CREDIT_SUM_LIMIT': ['mean', 'sum'],
        'AMT_ANNUITY': ['max', 'mean'],
        'DELAY_SERIOUSITY_SUM': ['mean', 'sum'],
        'STATUS_0_SUM': ['mean', 'sum'],
        'STATUS_1_SUM': ['mean', 'sum'],
        'STATUS_2_SUM': ['mean', 'sum'],
        'STATUS_3_SUM': ['mean', 'sum'],
        'STATUS_4_SUM': ['mean', 'sum'],
        'STATUS_5_SUM': ['mean', 'sum'],
        'STATUS_X_SUM': ['mean', 'sum'],
        'ACTIVE_MONTHS': ['mean', 'sum', 'max'],
        'STATUS_0_RATIO': ['mean'],
        'STATUS_1_RATIO': ['mean'],
        'STATUS_2_RATIO': ['mean'],
        'STATUS_3_RATIO': ['mean'],
        'STATUS_4_RATIO': ['mean'],
        'STATUS_5_RATIO': ['mean'],
        'STATUS_X_RATIO': ['mean'],
        'DELAYED_ONCE_OR_MORE': ['sum', 'mean'],
        'DLYED_OOM_RATIO': ['mean'],
        'WEIGHTED_DELAY_SUM': ['mean', 'sum'],
        'ACTIVE_LAST_MONTH': ['mean'],
        'BUREAU_LOAN_COUNT': ['mean'],
        'BUREAU_LOAN_TYPE': ['mean'],
        'ACTIVE_LOANS_PERCENTAGE': ['mean'],
        'CREDIT_ENDDATE_PERCENTAGE': ['mean'],
        'AVG_CREDITDAYS_PROLONGED': ['mean'],
        'AVERAGE_LOAN_TYPE': ['mean'],
        'DEBT_CREDIT_RATIO': ['mean'],
        'OVERDUE_DEBT_RATIO': ['mean'],
        'DAYS_DIFF': ['mean'],
        'DAYS_ENDDATE_DIFF': ['mean']
    }
    br_agg = br.groupby("SK_ID_CURR").agg(num_aggregations)
    br_agg.columns = pd.Index(['BURO_' + e[0] + '_' + e[1].upper()
                               for e in br_agg.columns.tolist()])

    active = br[br["CREDIT_ACTIVE"] == 'Active']
    active_agg = active.groupby("SK_ID_CURR").agg(num_aggregations)
    active_agg.columns = pd.Index(['ACTIVE_' + e[0] + '_' + e[1].upper()
                                   for e in active_agg.columns.tolist()])
    br_agg.join(active_agg, how='left', on='SK_ID_CURR')
    del active, active_agg
    gc.collect()

    closed = br[br["CREDIT_ACTIVE"] == "Closed"]
    closed_agg = closed.groupby("SK_ID_CURR").agg(num_aggregations)
    closed_agg.columns = pd.Index(['CLOSED_' + e[0] + '_' + e[1].upper()
                                   for e in closed_agg.columns.tolist()])
    br_agg = br_agg.join(closed_agg, how='left', on='SK_ID_CURR')
    del closed, closed_agg, br
    gc.collect()
    ipdb.set_trace()

    return br_agg


if __name__ == '__main__':
    bureau_and_balance()
