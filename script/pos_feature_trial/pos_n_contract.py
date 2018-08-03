import pandas as pd


def pos_cash(num_rows=None, nan_as_category=True):
    pos = pd.read_csv('../data/POS_CASH_balance.csv', nrows=num_rows)
    # SK_ID_CURR: size -- POSの支払い総回数
    # SK_ID_PREV: nunique -- POSの契約回数
    pos_agg = pos.groupby('SK_ID_CURR').agg({
        'SK_ID_CURR': 'size',
        'SK_ID_PREV': 'nunique'
    })
    pos_agg.columns = ['POS_COUNT', 'POS_N_CONTRACT']

    return pos_agg
