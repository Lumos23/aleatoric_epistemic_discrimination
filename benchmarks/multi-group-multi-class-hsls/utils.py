## standard packages
import sys
import numpy as np
import pandas as pd
import random
import pickle
from tqdm import tqdm
from time import localtime, strftime
import time

## scikitlearn
from sklearn import preprocessing
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
from sklearn.metrics import accuracy_score, log_loss, confusion_matrix, multilabel_confusion_matrix

from scipy.special import kl_div
from itertools import combinations

## custom packages
import coreMP as MP
import GroupFair as GF


def confusion(y, y_pred):
    TN, FP, FN, TP = confusion_matrix(y, y_pred).ravel()

    TPR = TP / (TP + FN)
    FPR = FP / (FP + TN)
    return TPR, FPR

def multilabel_confusion(y, y_pred, nc):
    print('calling multilabel_confusion')
    cm = multilabel_confusion_matrix(y, y_pred)  ## (nc, 2, 2) ##
    print('cm.shape:',cm.shape)
    print('y.shape:',y.shape,'y_pred.shape:', y_pred.shape)
    #     cm = confusion_matrix(y, y_pred)
    #     print(cm)
    tprs, fprs = np.zeros((nc)), np.zeros((nc))
    for i in range(nc):
        tn, fp, fn, tp = cm[i, :, :].ravel()

        tprs[i] = tp / (tp + fn)
        fprs[i] = fp / (fp + tn)
    return tprs, fprs

def odd_diffs_binary(y, y_pred, s):
    y0, y1 = y[s==1], y[s==2]
    y_pred0, y_pred1 = y_pred[s==1], y_pred[s==2]

    tpr0, fpr0 = confusion(y0, y_pred0)
    tpr1, fpr1 = confusion(y1, y_pred1)

    tpr_diff = tpr1 - tpr0
    fpr_diff = fpr1 - fpr0

    return (np.abs(tpr_diff) + np.abs(fpr_diff)) / 2, (np.abs(tpr_diff) + np.abs(fpr_diff)) / 2, max(np.abs(tpr_diff), np.abs(fpr_diff))

def odd_diffs_multi(y, y_pred, s, ns, nc):
    ## for binary case,
    tpr_diff, fpr_diff = np.zeros((nc, ns * (ns - 1) // 2)), np.zeros((nc, ns * (ns - 1) // 2))
    tprs, fprs = np.zeros((ns, nc)), np.zeros((ns, nc))
    print('s = ', s.shape, list(set(s)))
    for i in range(ns):
        y_s = y[s == i]
        y_pred_s = y_pred[s == i]
        tprs[i, :], fprs[i, :] = multilabel_confusion(y_s, y_pred_s, nc)

    for i in range(nc):
        tpr_diff[i, :] = np.array([a1 - a2 for (a1, a2) in combinations(tprs[:, i], 2)])
        fpr_diff[i, :] = np.array([a1 - a2 for (a1, a2) in combinations(fprs[:, i], 2)])


    meo = (np.abs(tpr_diff) + np.abs(fpr_diff) / (2 * ns) * nc).max()
    meo_abs = np.abs((tpr_diff + fpr_diff) / ns).max()
    mo = np.max(np.maximum(np.abs(tpr_diff), np.abs(fpr_diff)))
    return meo, meo_abs, mo

def oae_multi(y, y_pred, s, ns, nc):
    '''
    OAE for multiple groups and multiple labels.
    '''
    ## for binary case,
    tpr_diff, fpr_diff = np.zeros((nc, ns * (ns - 1) // 2)), np.zeros((nc, ns * (ns - 1) // 2))
    tprs, fprs = np.zeros((ns, nc)), np.zeros((ns, nc))
    print('s = ', s.shape, list(set(s)))
    for i in range(ns):
        y_s = y[s == i]
        y_pred_s = y_pred[s == i]
        tprs[i, :], fprs[i, :] = multilabel_confusion(y_s, y_pred_s, nc)

    for i in range(nc):
        tpr_diff[i, :] = np.array([a1 - a2 for (a1, a2) in combinations(tprs[:, i], 2)])
        fpr_diff[i, :] = np.array([a1 - a2 for (a1, a2) in combinations(fprs[:, i], 2)])


    meo = (np.abs(tpr_diff) + np.abs(fpr_diff) / (2 * ns) * nc).max()
    meo_abs = np.abs((tpr_diff + fpr_diff) / ns).max()
    mo = np.max(np.maximum(np.abs(tpr_diff), np.abs(fpr_diff)))
    return meo, meo_abs, mo

def oae_binary(y, y_pred, s):
    y0, y1 = y[s==0], y[s==1]
    y_pred0, y_pred1 = y_pred[s==0], y_pred[s==1]

    tpr0, fpr0 = confusion(y0, y_pred0)
    tpr1, fpr1 = confusion(y1, y_pred1)

    tpr_diff = tpr1 - tpr0


    return tpr_diff


def statistical_parity_binary(y, s):
    sp0 = y[s==0].mean()
    sp1 = y[s==1].mean()
    return np.abs(sp1-sp0)

def statistical_parity_multi(y, s, ns, nc):
    ## sp_{i, j} = y[s==i, y==j].sum() / len(y[s==i]) = Pr (Y= j |S =i), 1<=i<=ns, 1<=j<=nc
    sp = np.zeros((ns, nc))
    for i in range(ns):
        for j in range(nc):
            sp[i, j] = len(y[np.logical_and(s == i, y == j)]) / len(y[s == i])

    sp_class = []
    for j in range(nc):
        sp_class.append(max([np.abs(a1 - a2) for (a1, a2) in combinations(sp[:, j], 2)]))

    return max(sp_class) * (nc/ns)

def evaluation(idx1, idx2, clf, X, y, s, y_base, ns, nc,
               acc, kl, logloss, meo, meo_abs, mo, sp):
    y_prob = np.squeeze(clf.predict_proba(X=X, s=s), axis=2)
    y_pred = y_prob.argmax(axis=1)
    print("y:", y, "y_pred:",y_pred)
    acc[idx1, idx2] = accuracy_score(y, y_pred)
    kl[idx1, idx2] = kl_div(y_prob, y_base).mean()
    logloss[idx1, idx2] = kl_div(y_base, y_prob).mean()
    meo[idx1, idx2], meo_abs[idx1, idx2], mo[idx1, idx2] = odd_diffs_multi(y, y_pred, s, ns, nc)
    sp[idx1, idx2] = statistical_parity_multi(y_pred, s, ns, nc)

    return

def MP_tol(df, ns, nc, tolerance, use_protected, log, model='gbm', div='cross-entropy', num_iter=10, rand_seed=42, constraint='meo'):
    acc = np.zeros((len(tolerance), num_iter))
    kl = np.zeros((len(tolerance), num_iter))
    logloss = np.zeros((len(tolerance), num_iter))
    meo = np.zeros((len(tolerance), num_iter))
    meo_abs = np.zeros((len(tolerance), num_iter))
    mo = np.zeros((len(tolerance), num_iter))
    sp = np.zeros((len(tolerance), num_iter))
    dcp_msk = np.zeros((len(tolerance), num_iter))
    protected_attrs = ['racebin']
    label_name = 'grade'

    t_all = time.localtime()
    for seed in tqdm(range(num_iter)):
        log.write(' Iteration: {:2d}/{:2d}\n'.format(seed+1, num_iter))
        log.flush()
        t_epoch = time.localtime()
        ## train/test split using aif360.datasets.StandardDatasets
        df_train = df
        df_test = df.sample(frac=0.3, replace=True, random_state=seed)
        #df_test = df

        #df_train, df_test = train_test_split(df, test_size=0.3, random_state=seed)
        if use_protected:
            feature_names = list(set(df.columns.values) - set([label_name]))
        else:
            feature_names = list(set(df.columns.values) - set([label_name, protected_attrs[0]]))
        X_train, y_train = df_train.loc[:, feature_names].values, np.asarray(df_train[label_name].values.ravel())
        X_test, y_test = df_test.loc[:, feature_names].values, np.asarray(df_test[label_name].values.ravel())
        s_train, s_test = df_train[protected_attrs[0]].values.ravel(), df_test[protected_attrs[0]].values.ravel()

        # declare classifiers
        if model == 'gbm':
            clf_YgX = GradientBoostingClassifier(random_state=rand_seed)  # will predict Y from X
            clf_SgX = GradientBoostingClassifier(random_state=rand_seed)  # will predict S from X (needed for SP)
            clf_SgXY = GradientBoostingClassifier(random_state=rand_seed)  # will predict S from (X,Y)
        elif model == 'logit':
            clf_YgX = LogisticRegression(random_state=rand_seed)  # will predict Y from X
            clf_SgX = LogisticRegression(random_state=rand_seed)  # will predict S from X (needed for SP)
            clf_SgXY = LogisticRegression(random_state=rand_seed)  # will predict S from (X,Y)
        elif model == 'rfc':
            # for hsls 
            clf_YgX = RandomForestClassifier(n_estimators = 17, criterion = 'entropy', random_state = 42)
            clf_SgX = RandomForestClassifier(n_estimators = 17, criterion = 'entropy', random_state = 42)
            clf_SgXY = RandomForestClassifier( n_estimators = 17, criterion = 'entropy', random_state = 42)
            #clf_YgX = RandomForestClassifier(n_estimators = 100, criterion = 'entropy', random_state = 42)  # will predict Y from X
            # clf_SgX = RandomForestClassifier(random_state=rand_seed, n_estimators=100,warm_start = True,criterion = 'log_loss')
            # clf_SgXY = RandomForestClassifier(random_state=rand_seed, n_estimators=100,warm_start = True,criterion = 'log_loss')
        else:
            log.write('Error: Undefined Model\n')
            log.flush()
            return

        t_fit = time.localtime()
        ## initalize GFair class and train classifiers
        gf = GF.GFair(clf_YgX, clf_SgX, clf_SgXY, div=div)
        gf.fit(X=X_train, y=y_train, s=s_train, sample_weight=None)
        
        log.write('  Time to fit the base models: {:4.3f} mins\n'.format((time.mktime(time.localtime()) - time.mktime(t_fit))/60))
        log.flush()

        y_prob_base = clf_YgX.predict_proba(X_test)

        ## start projection
        for i, tol in enumerate(tolerance):
            t_tol = time.localtime()

            try: ## in case the solver has issues
                ## model projection
                constraints = [(constraint, tol)]
                gf.project(X=X_train, s=s_train, constraints=constraints, rho=2, max_iter=500, method='tf')

                log.write('  Tolerance: {:.4f}, projection time: {:4.3f} mins\n'.format(tol, (time.mktime(time.localtime()) - time.mktime(t_tol)) / 60))
                log.flush()

                ## evaluation
                print('beginning of evaluation')
                print(ns, nc, nc==5)
                print(list(set(y_test)))
                evaluation(i, seed, gf, X_test, y_test, s_test, y_prob_base, ns, nc,
                        acc, kl, logloss, meo, meo_abs, mo, sp)
                

            except:
                dcp_msk[i, seed] = 1
                log.write('  Tolerance: {:.4f}, Does not convergence!!!\n'.format(tol))
                log.flush()
                continue

        log.write('  Epoch Time: {:4.3f} mins\n'.format((time.mktime(time.localtime()) - time.mktime(t_epoch))/60))
        log.flush()

    results = {
        'acc': acc,
        'kl': kl,
        'logloss': logloss,
        'meo': meo,
        'meo_abs': meo_abs,
        'abseo': mo, # was originally named 'mo', changed to 'abseo' for consistency with other experiments
        'sp': sp,
        'dcp': dcp_msk
    }
    log.write(' Total Time: {:4.3f} mins\n'.format((time.mktime(time.localtime()) - time.mktime(t_all))/60))
    log.flush()
    return results
