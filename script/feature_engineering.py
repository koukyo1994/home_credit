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


def has_children_car_house(df):
    f = lambda x, car, house: 1 if (x["FLAG_OWN_CAR"] == car and
                                    x["FLAG_OWN_REALTY"] == house and
                                    x["CNT_CHILDREN"] > 0) else 0
    has_car_child = df.apply(lambda x: f(x, 'Y', 'Y'), axis=1) + \
                    df.apply(lambda x: f(x, 'Y', 'N'), axis=1)
    has_house_child = df.apply(lambda x: f(x, 'Y', 'Y'), axis=1) + \
                      df.apply(lambda x: f(x, 'N', 'Y'), axis=1)
    has_all = df.apply(lambda x: f(x, 'Y', 'Y'), axis=1)
    new_df = pd.DataFrame({
        "has_car_child": has_car_child,
        "has_house_child": has_house_child,
        "has_all": has_all
    })
    return new_df


def normalization_by(df, col, name):
    f = lambda x, arg1, support: \
        x[arg1] / support.loc[x[col, arg1]]
    inc_mean_by_region = df[["AMT_INCOME_TOTAL", col]] \
                         .groupby(col).mean()
    ann_mean_by_region = df[["AMT_ANNUITY", col]] \
                         .groupby(col).mean()
    cre_mean_by_region = df[["AMT_CREDIT", col]] \
                         .groupby(col).mean()
    goods_mean_by_region = df[["AMT_GOODS_PRICE", col]] \
                         .groupby(col).mean()
    inc_normalized = df.apply(
        lambda x: f(x, "AMT_INCOME_TOTAL", inc_mean_by_region),
        axis=1
    )
    ann_normalized = df.apply(
        lambda x: f(x, "AMT_ANNUITY", ann_mean_by_region),
        axis=1
    )
    cre_normalized = df.apply(
        lambda x: f(x, "AMT_CREDIT", cre_mean_by_region),
        axis=1
    )
    goods_normalized = df.apply(
        lambda x: f(x, "AMT_GOODS_PRICE", goods_normalized),
        axis=1
    )
    new_df = pd.DataFrame({
        f"income_normalized_by_{name}": inc_normalized,
        f"annuity_normalized_by_{name}": ann_normalized,
        f"credit_normalized_by_{name}": cre_normalized,
        f"goods_normalized_by_{name}": goods_normalized
    })
    return new_df


def apply_w_other_children(x):
    ret = 1 if (x["CNT_CHILDREN"] == 0 and
                x["NAME_TYPE_SUITE"] == "Children") else 0
    return ret


def application_train_test(num_rows = None, nan_as_category = False):
    # Read data and merge
    df = pd.read_csv('../data/application_train.csv', nrows= num_rows)
    test_df = pd.read_csv('../data/application_test.csv', nrows= num_rows)
    print(f"Train samples: {len(df)}, test samples: {len(test_df)}"
    df = df.append(test_df).reset_index()
    # Optional: Remove 4 applications with XNA CODE_GENDER (train set)
    df = df[df['CODE_GENDER'] != 'XNA']

    docs = [_f for _f in df.columns if 'FLAG_DOC' in _f]
    live = [_f for _f in df.columns if ('FLAG_' in _f) &
            ('FLAG_DOC' not in _f) & ('_FLAG_' not in _f)]

    # NaN values for DAYS_EMPLOYED: 365.243 -> nan
    df['DAYS_EMPLOYED'].replace(365243, np.nan, inplace= True)

    inc_by_org = df[['AMT_INCOME_TOTAL', 'ORGANIZATION_TYPE']].groupby(
        'ORGANIZATION_TYPE').median()['AMT_INCOME_TOTAL']

    df['NEW_CREDIT_TO_ANNUITY_RATIO'] = df['AMT_CREDIT'] / df['AMT_ANNUITY']
    df['NEW_CREDIT_TO_GOODS_RATIO'] = df['AMT_CREDIT'] / df['AMT_GOODS_PRICE']
    df["NEW_CREDIT_TO_INC_RATIO"] = df["AMT_CREDIT"] / df["AMT_INCOME_TOTAL"]
    df["NEW_CREDIT_TO_ANNUITY_RATIO"] = df["AMT_CREDIT"] / df["AMT_ANNUITY"]
    df["NEW_APPLY_W_OTHER_CHILDREN"] = df.apply(
        lambda x: apply_w_other_children(x),
        axis=1
    )
    df['NEW_DOC_IND_KURT'] = df[docs].kurtosis(axis=1)
    df['NEW_LIVE_IND_SUM'] = df[live].sum(axis=1)
    df['NEW_INC_CHI'] = df['AMT_INCOME_TOTAL'] / (1 + df["CNT_CHILDREN"])
    df['NEW_INC_BY_ORG'] = df['ORGANIZATION_TYPE'].map(inc_by_org)
    df['NEW_EMPLOY_TO_BIRTH_RATIO'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']
    df['NEW_ANNUITY_TO_INCOME_RATIO'] = df['AMT_ANNUITY'] / (1 + df['AMT_INCOME_TOTAL'])
    df['NEW_SOURCES_PROD'] = df['EXT_SOURCE_1'] * df['EXT_SOURCE_2'] * df['EXT_SOURCE_3']
    df['NEW_EXT_SOURCES_MEAN'] = df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].mean(axis=1)
    df['NEW_SCORES_STD'] = df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].std(axis=1)
    df['NEW_SCORES_STD'] = df['NEW_SCORES_STD'].fillna(df['NEW_SCORES_STD'].mean())
    df['NEW_CAR_TO_BIRTH_RATIO'] = df['OWN_CAR_AGE'] / df['DAYS_BIRTH']
    df['NEW_CAR_TO_EMPLOY_RATIO'] = df['OWN_CAR_AGE'] / df['DAYS_EMPLOYED']
    df['NEW_PHONE_TO_BIRTH_RATIO'] = df['DAYS_LAST_PHONE_CHANGE'] / df['DAYS_BIRTH']
    df['NEW_PHONE_TO_BIRTH_RATIO_EMPLOYER'] = df['DAYS_LAST_PHONE_CHANGE'] / df['DAYS_EMPLOYED']
    df['NEW_CREDIT_TO_INCOME_RATIO'] = df['AMT_CREDIT'] / df['AMT_INCOME_TOTAL']
    df["NEW_ANNUAL_PAYMENT"] = df["AMT_CREDIT"] / (365*100 + df["DAYS_BIRTH"])
    df["NEW_ANNUAL_PERCENT"] = df["AMT_INCOME_TOTAL"] / df["NEW_ANNUAL_PAYMENT"]
    df["NDOC"] = 0
    for i in range(2, 22):
        df["NDOC"] += df[f"FLAG_DOCUMENT_{i}"]

    # add custom features
    car_owner_gen = car_owner_w_gender(df)
    realty_own_gen = realty_own_w_gender(df)
    child_w_gen = has_children_w_gender(df)
    many_children_gen = has_many_children_w_gender(df, 4)
    car_house_children = has_children_car_house(df)
    normalized_by_region = normalization_by(df,
                                            "REGION_POPULATION_RELATIVE",
                                            "region")
    normalized_by_inc_type = normalization_by(df,
                                              "NAME_INCOME_TYPE",
                                              "inctype")
    normalized_by_edu = normalization_by(df,
                                         "NAME_EDUCATION_TYPE",
                                         "edutype")
    normalized_by_housing = normalization_by(df,
                                             "NAME_HOUSING_TYPE",
                                             "housing")
    normalized_by_city = normalization_by(df,
                                          "REGION_RATING_CLIENT_W_CITY",
                                          "cityrating")
    normalized_by_rating = normalization_by(df,
                                            "REGION_RATING_CLIENT",
                                            "rating")
    normalized_by_org = normalization_by(df,
                                         "ORGANIZATION_TYPE",
                                         "org")

    df = pd.concat([df, car_owner_gen, realty_own_gen, child_w_gen,
                    many_children_gen, car_house_children,
                    normalized_by_region, normalized_by_inc_type,
                    normalized_by_edu, normalized_by_housing,
                    normalized_by_city, normalized_by_rating,
                    normalized_by_org], axis=1)

    # Categorical features with Binary encode (0 or 1; two categories)
    for bin_feature in ['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY']:
        df[bin_feature], uniques = pd.factorize(df[bin_feature])

    # Categorical features with One-Hot encode
    df, cat_cols = one_hot_encoder(df, nan_as_category)

    dropcolum=['FLAG_DOCUMENT_2','FLAG_DOCUMENT_4',
    'FLAG_DOCUMENT_5','FLAG_DOCUMENT_6','FLAG_DOCUMENT_7',
    'FLAG_DOCUMENT_8','FLAG_DOCUMENT_9','FLAG_DOCUMENT_10',
    'FLAG_DOCUMENT_11','FLAG_DOCUMENT_12','FLAG_DOCUMENT_13',
    'FLAG_DOCUMENT_14','FLAG_DOCUMENT_15','FLAG_DOCUMENT_16',
    'FLAG_DOCUMENT_17','FLAG_DOCUMENT_18','FLAG_DOCUMENT_19',
    'FLAG_DOCUMENT_20','FLAG_DOCUMENT_21']
    df= df.drop(dropcolum,axis=1)
    del test_df
    gc.collect()
    return df


def bureau_and_balance(num_rows = None, nan_as_category = True):
    bureau = pd.read_csv('../data/bureau.csv', nrows = num_rows)
    bb = pd.read_csv('../data/bureau_balance.csv', nrows = num_rows)
    bb, bb_cat = one_hot_encoder(bb, nan_as_category)
    bureau, bureau_cat = one_hot_encoder(bureau, nan_as_category)

    # Bureau balance: Perform aggregations and merge with bureau.csv
    bb_aggregations = {'MONTHS_BALANCE': ['min', 'max', 'size']}
    for col in bb_cat:
        bb_aggregations[col] = ['mean']
    bb_agg = bb.groupby('SK_ID_BUREAU').agg(bb_aggregations)
    bb_agg.columns = pd.Index([e[0] + "_" + e[1].upper() for e in bb_agg.columns.tolist()])
    bureau = bureau.join(bb_agg, how='left', on='SK_ID_BUREAU')
    bureau.drop(['SK_ID_BUREAU'], axis=1, inplace= True)
    del bb, bb_agg
    gc.collect()

    # Bureau and bureau_balance numeric features
    num_aggregations = {
        'DAYS_CREDIT': [ 'mean', 'var'],
        'DAYS_CREDIT_ENDDATE': [ 'mean'],
        'DAYS_CREDIT_UPDATE': ['mean'],
        'CREDIT_DAY_OVERDUE': ['mean'],
        'AMT_CREDIT_MAX_OVERDUE': ['mean'],
        'AMT_CREDIT_SUM': [ 'mean', 'sum'],
        'AMT_CREDIT_SUM_DEBT': [ 'mean', 'sum'],
        'AMT_CREDIT_SUM_OVERDUE': ['mean'],
        'AMT_CREDIT_SUM_LIMIT': ['mean', 'sum'],
        'AMT_ANNUITY': ['max', 'mean'],
        'CNT_CREDIT_PROLONG': ['sum'],
        'MONTHS_BALANCE_MIN': ['min'],
        'MONTHS_BALANCE_MAX': ['max'],
        'MONTHS_BALANCE_SIZE': ['mean', 'sum']
    }
    # Bureau and bureau_balance categorical features
    cat_aggregations = {}
    for cat in bureau_cat: cat_aggregations[cat] = ['mean']
    for cat in bb_cat: cat_aggregations[cat + "_MEAN"] = ['mean']

    bureau_agg = bureau.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
    bureau_agg.columns = pd.Index(['BURO_' + e[0] + "_" + e[1].upper() for e in bureau_agg.columns.tolist()])
    # Bureau: Active credits - using only numerical aggregations
    active = bureau[bureau['CREDIT_ACTIVE_Active'] == 1]
    active_agg = active.groupby('SK_ID_CURR').agg(num_aggregations)
    active_agg.columns = pd.Index(['ACTIVE_' + e[0] + "_" + e[1].upper() for e in active_agg.columns.tolist()])
    bureau_agg = bureau_agg.join(active_agg, how='left', on='SK_ID_CURR')
    del active, active_agg
    gc.collect()
    # Bureau: Closed credits - using only numerical aggregations
    closed = bureau[bureau['CREDIT_ACTIVE_Closed'] == 1]
    closed_agg = closed.groupby('SK_ID_CURR').agg(num_aggregations)
    closed_agg.columns = pd.Index(['CLOSED_' + e[0] + "_" + e[1].upper() for e in closed_agg.columns.tolist()])
    bureau_agg = bureau_agg.join(closed_agg, how='left', on='SK_ID_CURR')
    del closed, closed_agg, bureau
    gc.collect()
    return bureau_agg
