import numpy as np
import cvxpy as cp
import dccp

import random
from numpy.random import choice
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="darkgrid")

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.decomposition import PCA
import matplotlib.patches as mpatches
import random
import pickle
import time
from tqdm import tqdm
from time import localtime, strftime

from IPython.display import Markdown, display
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression 
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import train_test_split

# 'synth2':
def sample_from_gaussian(pos_mean,
                            pos_cov,
                            neg_mean,
                            neg_cov,
                            thr=0,
                            n_pos=200,
                            n_neg=200,
                            seed=0,
                            corr_sens=True):
    '''
    generate Gaussian dataset
    '''
    np.random.seed(seed)
    x_pos = np.random.multivariate_normal(pos_mean, pos_cov, n_pos)
    np.random.seed(seed)
    x_neg = np.random.multivariate_normal(neg_mean, neg_cov, n_neg)
    X = np.vstack((x_pos, x_neg))
    y = np.hstack((np.ones(n_pos), np.zeros(n_neg)))
    n = y.shape[0]
    if corr_sens:
        # correlated sens data
        sens_attr = np.zeros(n)
        idx = np.where(X[:,0] > thr)[0]
        sens_attr[idx] = 1
    else:
        # independent sens data
        np.random.seed(seed)
        sens_attr = np.random.binomial(1, 0.5, n)
    return X, y, sens_attr

## NOTE change these variables for different distribution/generation of synth data.
pos_mean = np.array([2,2])
pos_cov = np.array([[5, 1], [1,5]])
neg_mean = np.array([-2,-2])
neg_cov = np.array([[10, 1],[1, 3]])
n_pos = 1000
n_neg = 600
thr = 0
corr_sens = True
X, y, sens = sample_from_gaussian(pos_mean,
                                    pos_cov,
                                    neg_mean,
                                    neg_cov,
                                    thr=thr,
                                    n_pos=n_pos,
                                    n_neg=n_neg,
                                    corr_sens=corr_sens)
X = np.concatenate((X, np.expand_dims(sens, 1)), axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=seed)
dtypes = None
dtypes_ = None
sens_idc = [2]
X_train_removed = X_train[:,:2]
X_test_removed = X_test[:,:2]
race_idx = None
sex_idx = None

df_array = np.hstack([X_train, y_train.reshape(-1, 1)])
df = pd.DataFrame(df_array)
df = df.rename(columns={0: "x0", 1: "x1", 2: "s", 3: "y"})

df.to_pickle("./gaussian_synth2.pkl")  