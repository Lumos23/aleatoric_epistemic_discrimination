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
from utils_with_sensitive import MP_tol
from DataLoader import *


def main(argv):
    df = load_data(name='adult', modified = True)
    repetition = 10
    use_protected = True
    use_sample_weight = True
    tune_threshold = True

    # adult tolerance
    tolerance = [0.07, 0.075, 0.08, 0.085, 0.09, 0.095, 0.1, 0.5, 0.75, 1.0] 

    # COMPAS tolerance
    #tolerance = [0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.1, 0.5, 1.0]

    # German Credits tolerance
    #tolerance = [0.005, 0.01, 0.02, 0.07, 0.1] 

    try:
        opts, args = getopt.getopt(argv,"hm:i:r:s:")
    except getopt.GetoptError:
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('Correct Usage:\n')
            print('python run-mp.py -i [inputfile] -r [repetition] -s [seed]')
            print('\n')
            print('Options for arguments:')
            print('[inputfile]: adult, compas, hsls ...  (Default: adult)')
            print('[reptition]: number of iterations to run for each point (Default: 10)')
            print('\n')
            sys.exit()
        elif opt == '-r':
            repetition = int(arg)
        elif opt == '-i':
            inputfile = arg
            if inputfile == 'adult':
                df = load_data('adult', modified = True, perturbed= False)
                protected_attrs = ['gender']
                label_name = 'income'
            elif inputfile == 'compas':
                df = load_data('compas', modified = True, perturbed= False)
                protected_attrs = ['race']
                label_name = 'is_recid'

            elif inputfile == 'german_credit':
                with open('../data/german_credit/german_data_processed.pkl', 'rb') as file:
                    df = pickle.load(file)
                df.drop('SY',inplace=True,axis=1) #drop the "SY" column from input
                protected_attrs = ['age']
                label_name = 'credit'
            else: 
                print('Invalid Input Dataset Name')
                sys.exit(2)

    start_time = time.localtime()
    start_time_str = strftime("%Y-%m-%d-%H.%M.%S", start_time)
    filename = inputfile+'-'+ str(df.shape[0]) +'-mp-' + start_time_str
    f = open(filename+'-log.txt','w')

    f.write('Setup Summary\n')
    f.write(' Sampled Dataset Shape: ' + str(df.shape) + '\n')
    f.write(' repetition: '+str(repetition) + '\n')
    f.write(' use_protected: '+str(use_protected) + '\n')
    f.write(' use_sample_weight: '+str(use_sample_weight) + '\n')
    f.write(' tune_threshold: '+str(tune_threshold) + '\n')
    f.write(' tolerance: '+str(tolerance) + '\n')
    f.write('adult dataset with original baseline'+ '\n')
    f.flush()

 
    # Random Forest
    f.write('RFC - CE - meo\n')
    tune_threshold = True
    rfc_ce_meo = MP_tol(df, protected_attrs=protected_attrs, label_name=label_name, use_protected = use_protected, use_sample_weight=use_sample_weight, tune_threshold=tune_threshold, tolerance=tolerance, log = f, model='rfc', div='cross-entropy', num_iter=repetition, rand_seed=42, constraint='meo')
   
    save = {
        'rfc_ce_meo': rfc_ce_meo,
        'tolerance': tolerance
    }


    result_path = './results/'
    savename = inputfile+'_fairprojection.pkl'
    
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
