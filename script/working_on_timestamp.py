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
