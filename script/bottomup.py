import gc
import os
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.io import savemat, loadmat
from scipy.sparse import csr_matrix, csc_matrix, hstack
from contextlib import contextmanager
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold
from lightgbm import LGBMClassifier
warnings.simplefilter(action="ignore", category=FutureWarning)


class TargetEncoder:
    def __init__(self):
        self.encoder = None

    def fit(self, cat, target):
        df = pd.concat([cat, target], axis=1)
        colname_category = cat.name
        colname_target = target.name

        self.encoder = df.groupby(colname_category)[colname_target].mean()

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


def one_hot_encoder(df, nan_as_category=True, sparse=True):
    columns = df.columns
    categoricals = [col for col in columns if df[col].dtype == "object"]
    new_df = pd.get_dummies(df,
                            columns=categoricals,
                            dummy_na=nan_as_category)
    new_columns = [col for col in new_df.columns]
    return new_df, new_columns


def pos_cash(num_rows=None, nan_as_category=True):
    pos = pd.read_csv('../data/POS_CASH_balance.csv', nrows=num_rows)

    # SK_ID_CURR: size -- POSの支払い総回数
    # SK_ID_PREV: nunique -- POSの契約回数
    pos_agg_gb_curr = pos.groupby('SK_ID_CURR').agg({'SK_ID_CURR': 'size'})
    pos_agg_gb_curr.columns = ['POS_COUNT']

    pos_agg_gb_prev = pos.groupby('SK_ID_PREV').agg({
        'SK_ID_CURR': 'max',
        'CNT_INSTALMENT': ['max', 'min']
    })
    pos_agg_gb_prev.columns = [e[0] + '_' + e[1].upper()
                               for e in pos_agg_gb_prev.columns]
    pos_agg_gb_prev = pos_agg_gb_prev.rename(columns={
        'SK_ID_CURR_MAX': 'SK_ID_CURR'
    })
    pos_agg_gb_prev['POS_CHANGED_CONTRACT'] = \
    (pos_agg_gb_prev['CNT_INSTALMENT_MAX'] != pos_agg_gb_prev['CNT_INSTALMENT_MIN']) \
    .astype(int)
    del pos_agg_gb_prev['CNT_INSTALMENT_MAX'], pos_agg_gb_prev['CNT_INSTALMENT_MIN']
    pos_agg = pos_agg_gb_curr.merge(pos_agg_gb_prev,
                                    how='left',
                                    on='SK_ID_CURR') \
        .groupby('SK_ID_CURR').agg({
            'POS_COUNT': 'mean',
            'POS_CHANGED_CONTRACT': 'mean'
        })
    return pos_agg


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
            nthread=8,
            #is_unbalance=True,
            n_estimators=10000,
            learning_rate=0.1,
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


def display_importances(feature_importance_df_):
    cols = feature_importance_df_[["feature", "importance"]].groupby("feature").mean().sort_values(by="importance", ascending=False)[:40].index
    best_features = feature_importance_df_.loc[feature_importance_df_.feature.isin(cols)]
    plt.figure(figsize=(8, 10))
    sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False))
    plt.title('LightGBM Features (avg over folds)')
    plt.tight_layout()
    plt.savefig(figure_name)


def main(debug=False):
    num_rows = 10000 if debug else None
    with timer('Process base application train test features'):
        df = application_train_test(num_rows)
        print('application_train_test shape: ', df.shape)
    with timer('Process POS_CASH balance'):
        pos = pos_cash(num_rows)
        print('Pos_cash shape: ', pos.shape)
        df = df.merge(pos, how='left', on='SK_ID_CURR')
        del pos
        gc.collect()
    with timer('Run LightGBM with KFold'):
        feat_importance = kfold_lightgbm(df,
                                         num_folds=5,
                                         stratified=False,
                                         debug=debug)


if __name__ == '__main__':
    filename = "../memo/log/bottom_up_pos_changed_contract.txt"
    submission_file_name = "../submission/kernel_bottomup_pos_changed_contract.csv"
    figure_name = "../figure/bottom_up_pos_changed_contract.png"
    with timer("Full model run"):
        main()
