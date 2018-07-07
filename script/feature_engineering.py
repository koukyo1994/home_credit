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
