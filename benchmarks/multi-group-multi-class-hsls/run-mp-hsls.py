## standard packages
import sys
import os.path
import numpy as np
import pandas as pd
import random
import pickle
from tqdm import tqdm
from time import localtime, strftime
import time
import argparse
import getopt

## scikit learn
from sklearn import preprocessing
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

## aif360
from aif360.datasets import StandardDataset

## custom packages
from utils import MP_tol


def main(argv):
    df = pd.read_pickle("../data/HSLS/hsls_discretized_multis.pkl")  
    df = df[['X1MTHID', 'X1MTHUTI', 'X1MTHEFF','X1PAR1EDU',
       'X1FAMINCOME', 'X1SCHOOLBEL', 'racebin',
       'sexbin', 'grade']]
    print('Processed Dataset Shape:', df.shape)
    repetition = 1
    use_protected = True
    multigroup = True
    n_classes = 5
    n_groups = 4
    # HSLS tolerance
    tolerance = [0.06, 0.065, 0.07, 0.08, 0.1, 0.09, 0.15, 0.2, 0.3]

    try:
        opts, args = getopt.getopt(argv,"hm:i:r:s:")
    except getopt.GetoptError:
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('Correct Usage:\n')
            print('python run-mp.py -mg [multigroup] -nc [n_classes] -ng [n_groups] -r [repetition] ')
            sys.exit()
        elif opt == '-r':
            repetition = int(arg)
        elif opt == '-mg':
            multigroup = arg
        elif opt == '-nc':
            n_classes = int(arg)
        elif opt == 'ng':
            n_groups == int(arg)

    start_time = time.localtime()
    start_time_str = strftime("%Y-%m-%d-%H.%M.%S", start_time)
    filename = 'hsls'+'-'+ str(df.shape[0]) +'-mp-' + start_time_str
    f = open(filename+'-log.txt','w')

    f.write('Setup Summary\n')
    f.write(' Sampled Dataset Shape: ' + str(df.shape) + '\n')
    f.write(' repetition: '+str(repetition) + '\n')
    f.write(' use_protected: '+str(use_protected) + '\n')
    f.write(' n_classes: ' + str(n_classes) + '\n')
    f.write(' n_groups: ' + str(n_groups) + '\n')
    f.write(' multigroup: '+str(multigroup) + '\n')
    f.write(' tolerance: '+str(tolerance) + '\n')
    f.write('n_estimator = 17')
    f.flush()

 
    # Random Forest
    f.write('RFC - CE - meo\n')
    #rfc_ce_meo = MP_tol(df, ns=n_groups, nc=n_classes, use_protected = use_protected, tolerance=tolerance, log = f, model='rfc', div='cross-entropy', num_iter=repetition, rand_seed=42, constraint='meo')
    rf_ce_sp = MP_tol(df, ns=n_groups, nc=n_classes, use_protected = use_protected, tolerance=tolerance, log = f, model='rfc', div='cross-entropy', num_iter=repetition, rand_seed=42, constraint='sp')

    save = {
        # 'rfc_ce_meo': rfc_ce_meo,
        'rf_ce_sp': rf_ce_sp,
        'tolerance': tolerance
    }


    result_path = './'
    savename = 'hsls_fairprojection_estimator17_10run_sp.pkl'
    
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    with open(result_path+savename, 'wb+') as pickle_f:
        pickle.dump(save, pickle_f, 2)

    f.write('Total Run Time: {:4.3f} mins\n'.format((time.mktime(time.localtime()) - time.mktime(start_time))/60))
    f.write('Finished!!!\n')
    f.flush()
    f.close()
    



if __name__ == "__main__":
    main(sys.argv[1:])
