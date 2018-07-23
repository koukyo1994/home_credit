import gc
import os
import time
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


def application_train_test(num_rows = None, nan_as_category = False):
    # Read data and merge
    df = pd.read_csv('../data/application_train.csv', nrows= num_rows)
    test_df = pd.read_csv('../data/application_test.csv', nrows= num_rows)
    print(f"Train samples: {len(df)}, test samples: {len(test_df)}")
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

    return br_agg


# Preprocess previous_applications.csv
def previous_applications(num_rows = None, nan_as_category = True):
    prev = pd.read_csv('../data/previous_application.csv', nrows = num_rows)
    prev, cat_cols = one_hot_encoder(prev, nan_as_category= True)
    # Days 365.243 values -> nan
    prev['DAYS_FIRST_DRAWING'].replace(365243, np.nan, inplace= True)
    prev['DAYS_FIRST_DUE'].replace(365243, np.nan, inplace= True)
    prev['DAYS_LAST_DUE_1ST_VERSION'].replace(365243, np.nan, inplace= True)
    prev['DAYS_LAST_DUE'].replace(365243, np.nan, inplace= True)
    prev['DAYS_TERMINATION'].replace(365243, np.nan, inplace= True)
    # Add feature: value ask / value received percentage
    prev['APP_CREDIT_PERC'] = prev['AMT_APPLICATION'] / prev['AMT_CREDIT']
    # Previous applications numeric features
    num_aggregations = {
        'AMT_ANNUITY': [ 'max', 'mean'],
        'AMT_APPLICATION': [ 'max','mean'],
        'AMT_CREDIT': [ 'max', 'mean'],
        'APP_CREDIT_PERC': [ 'max', 'mean'],
        'AMT_DOWN_PAYMENT': [ 'max', 'mean'],
        'AMT_GOODS_PRICE': [ 'max', 'mean'],
        'HOUR_APPR_PROCESS_START': [ 'max', 'mean'],
        'RATE_DOWN_PAYMENT': [ 'max', 'mean'],
        'DAYS_DECISION': [ 'max', 'mean'],
        'CNT_PAYMENT': ['mean', 'sum'],
    }
    # Previous applications categorical features
    cat_aggregations = {}
    for cat in cat_cols:
        cat_aggregations[cat] = ['mean']

    prev_agg = prev.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
    prev_agg.columns = pd.Index(['PREV_' + e[0] + "_" + e[1].upper() for e in prev_agg.columns.tolist()])
    # Previous Applications: Approved Applications - only numerical features
    approved = prev[prev['NAME_CONTRACT_STATUS_Approved'] == 1]
    approved_agg = approved.groupby('SK_ID_CURR').agg(num_aggregations)
    approved_agg.columns = pd.Index(['APPROVED_' + e[0] + "_" + e[1].upper() for e in approved_agg.columns.tolist()])
    prev_agg = prev_agg.join(approved_agg, how='left', on='SK_ID_CURR')
    # Previous Applications: Refused Applications - only numerical features
    refused = prev[prev['NAME_CONTRACT_STATUS_Refused'] == 1]
    refused_agg = refused.groupby('SK_ID_CURR').agg(num_aggregations)
    refused_agg.columns = pd.Index(['REFUSED_' + e[0] + "_" + e[1].upper() for e in refused_agg.columns.tolist()])
    prev_agg = prev_agg.join(refused_agg, how='left', on='SK_ID_CURR')
    del refused, refused_agg, approved, approved_agg, prev
    gc.collect()
    return prev_agg

# Preprocess POS_CASH_balance.csv
def pos_cash(num_rows = None, nan_as_category = True):
    pos = pd.read_csv('../data/POS_CASH_balance.csv', nrows = num_rows)
    pos, cat_cols = one_hot_encoder(pos, nan_as_category= True)
    # Features
    aggregations = {
        'MONTHS_BALANCE': ['max', 'mean', 'size'],
        'SK_DPD': ['max', 'mean'],
        'SK_DPD_DEF': ['max', 'mean']
    }
    for cat in cat_cols:
        aggregations[cat] = ['mean']

    pos_agg = pos.groupby('SK_ID_CURR').agg(aggregations)
    pos_agg.columns = pd.Index(['POS_' + e[0] + "_" + e[1].upper() for e in pos_agg.columns.tolist()])
    # Count pos cash accounts
    pos_agg['POS_COUNT'] = pos.groupby('SK_ID_CURR').size()
    del pos
    gc.collect()
    return pos_agg

# Preprocess installments_payments.csv
def installments_payments(num_rows = None, nan_as_category = True):
    ins = pd.read_csv('../data/installments_payments.csv', nrows = num_rows)
    ins, cat_cols = one_hot_encoder(ins, nan_as_category= True)
    # Percentage and difference paid in each installment (amount paid and installment value)
    ins['PAYMENT_PERC'] = ins['AMT_PAYMENT'] / ins['AMT_INSTALMENT']
    ins['PAYMENT_DIFF'] = ins['AMT_INSTALMENT'] - ins['AMT_PAYMENT']
    # Days past due and days before due (no negative values)
    ins['DPD'] = ins['DAYS_ENTRY_PAYMENT'] - ins['DAYS_INSTALMENT']
    ins['DBD'] = ins['DAYS_INSTALMENT'] - ins['DAYS_ENTRY_PAYMENT']
    ins['DPD'] = ins['DPD'].apply(lambda x: x if x > 0 else 0)
    ins['DBD'] = ins['DBD'].apply(lambda x: x if x > 0 else 0)
    # Features: Perform aggregations
    aggregations = {
        'NUM_INSTALMENT_VERSION': ['nunique'],
        'DPD': ['max', 'mean', 'sum'],
        'DBD': ['max', 'mean', 'sum'],
        'PAYMENT_PERC': [ 'mean',  'var'],
        'PAYMENT_DIFF': [ 'mean', 'var'],
        'AMT_INSTALMENT': ['max', 'mean', 'sum'],
        'AMT_PAYMENT': ['min', 'max', 'mean', 'sum'],
        'DAYS_ENTRY_PAYMENT': ['max', 'mean', 'sum']
    }
    for cat in cat_cols:
        aggregations[cat] = ['mean']
    ins_agg = ins.groupby('SK_ID_CURR').agg(aggregations)
    ins_agg.columns = pd.Index(['INSTAL_' + e[0] + "_" + e[1].upper() for e in ins_agg.columns.tolist()])
    # Count installments accounts
    ins_agg['INSTAL_COUNT'] = ins.groupby('SK_ID_CURR').size()
    del ins
    gc.collect()
    return ins_agg

# Preprocess credit_card_balance.csv
def credit_card_balance(num_rows = None, nan_as_category = True):
    cc = pd.read_csv('../data/credit_card_balance.csv', nrows = num_rows)
    cc, cat_cols = one_hot_encoder(cc, nan_as_category= True)
    # General aggregations
    cc.drop(['SK_ID_PREV'], axis= 1, inplace = True)
    cc_agg = cc.groupby('SK_ID_CURR').agg([ 'max', 'mean', 'sum', 'var'])
    cc_agg.columns = pd.Index(['CC_' + e[0] + "_" + e[1].upper() for e in cc_agg.columns.tolist()])
    # Count credit card lines
    cc_agg['CC_COUNT'] = cc.groupby('SK_ID_CURR').size()
    del cc
    gc.collect()
    return cc_agg

# LightGBM GBDT with KFold or Stratified KFold
# Parameters from Tilii kernel: https://www.kaggle.com/tilii7/olivier-lightgbm-parameters-by-bayesian-opt/code
def kfold_lightgbm(df, num_folds, stratified = False, debug= False):
    # Divide in training/validation and test data
    train_df = df[df['TARGET'].notnull()]
    test_df = df[df['TARGET'].isnull()]
    print("Starting LightGBM. Train shape: {}, test shape: {}".format(train_df.shape, test_df.shape))
    del df
    gc.collect()
    # Cross validation model
    if stratified:
        folds = StratifiedKFold(n_splits= num_folds, shuffle=True, random_state=47)
    else:
        folds = KFold(n_splits= num_folds, shuffle=True, random_state=47)
    # Create arrays and dataframes to store results
    oof_preds = np.zeros(train_df.shape[0])
    sub_preds = np.zeros(test_df.shape[0])
    feature_importance_df = pd.DataFrame()
    feats = [f for f in train_df.columns if f not in ['TARGET','SK_ID_CURR','SK_ID_BUREAU','SK_ID_PREV','index']]

    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df[feats], train_df['TARGET'])):
        train_x, train_y = train_df[feats].iloc[train_idx], train_df['TARGET'].iloc[train_idx]
        valid_x, valid_y = train_df[feats].iloc[valid_idx], train_df['TARGET'].iloc[valid_idx]

        # LightGBM parameters found by Bayesian optimization
        clf = LGBMClassifier(
            nthread=4,
            #is_unbalance=True,
            n_estimators=10000,
            learning_rate=0.02,
            num_leaves=32,
            colsample_bytree=0.9497036,
            subsample=0.8715623,
            max_depth=8,
            reg_alpha=0.04,
            reg_lambda=0.073,
            min_split_gain=0.0222415,
            min_child_weight=40,
            silent=-1,
            verbose=-1,
            #scale_pos_weight=11
            )

        clf.fit(train_x, train_y, eval_set=[(train_x, train_y), (valid_x, valid_y)],
            eval_metric= 'auc', verbose= 100, early_stopping_rounds= 200)

        oof_preds[valid_idx] = clf.predict_proba(valid_x, num_iteration=clf.best_iteration_)[:, 1]
        sub_preds += clf.predict_proba(test_df[feats], num_iteration=clf.best_iteration_)[:, 1] / folds.n_splits

        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = feats
        fold_importance_df["importance"] = clf.feature_importances_
        fold_importance_df["fold"] = n_fold + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
        print('Fold %2d AUC : %.6f' % (n_fold + 1, roc_auc_score(valid_y, oof_preds[valid_idx])))
        with open(filename, 'a') as f:
            f.write(f"Fold {n_fold+1} AUC: {roc_auc_score(valid_y, oof_preds[valid_idx]):.6f}\n")
        del clf, train_x, train_y, valid_x, valid_y
        gc.collect()


    print('Full AUC score %.6f' % roc_auc_score(train_df['TARGET'], oof_preds))
    with open(filename, 'a') as f:
        f.write(f"Full AUC: {roc_auc_score(train_df['TARGET'], oof_preds):.6f}\n")
    # Write submission file and plot feature importance
    if not debug:
        test_df['TARGET'] = sub_preds
        test_df[['SK_ID_CURR', 'TARGET']].to_csv(submission_file_name, index= False)
    display_importances(feature_importance_df)
    return feature_importance_df

# Display/plot feature importance
def display_importances(feature_importance_df_):
    cols = feature_importance_df_[["feature", "importance"]].groupby("feature").mean().sort_values(by="importance", ascending=False)[:40].index
    best_features = feature_importance_df_.loc[feature_importance_df_.feature.isin(cols)]
    plt.figure(figsize=(8, 10))
    sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False))
    plt.title('LightGBM Features (avg over folds)')
    plt.tight_layout()
    plt.savefig('../figure/lgbm_importances03.png')


def main(debug = False):
    num_rows = 10000 if debug else None
    with timer("Process base application train test features"):
        df = application_train_test(num_rows)
        print("application_train_test shape: ", df.shape)
    with timer("Process bureau and bureau_balance"):
        bureau = bureau_and_balance(num_rows)
        print("Bureau df shape:", bureau.shape)
        df = df.join(bureau, how='left', on='SK_ID_CURR')
        del bureau
        gc.collect()
    with timer("Process previous_applications"):
        prev = previous_applications(num_rows)
        print("Previous applications df shape:", prev.shape)
        df = df.join(prev, how='left', on='SK_ID_CURR')
        del prev
        gc.collect()
    with timer("Process POS-CASH balance"):
        pos = pos_cash(num_rows)
        print("Pos-cash balance df shape:", pos.shape)
        df = df.join(pos, how='left', on='SK_ID_CURR')
        del pos
        gc.collect()
    with timer("Process installments payments"):
        ins = installments_payments(num_rows)
        print("Installments payments df shape:", ins.shape)
        df = df.join(ins, how='left', on='SK_ID_CURR')
        del ins
        gc.collect()
    with timer("Process credit card balance"):
        cc = credit_card_balance(num_rows)
        print("Credit card balance df shape:", cc.shape)
        df = df.join(cc, how='left', on='SK_ID_CURR')
        del cc
        gc.collect()
    with timer("Run LightGBM with kfold"):
        feat_importance = kfold_lightgbm(df, num_folds= 5, stratified= False, debug= debug)

if __name__ == "__main__":
    filename = "../memo/log/learning_log03.txt"
    submission_file_name = "../submission/kernel03.csv"
    with timer("Full model run"):
        main()
