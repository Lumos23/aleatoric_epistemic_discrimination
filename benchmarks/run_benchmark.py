#!/usr/bin/python

import sys
import os
import getopt
import pickle
import pandas as pd

from benchmark import Benchmark
from DataLoader import *

from utils_with_sensitive import leveraging_approach

import warnings
warnings.filterwarnings("ignore")


def main(argv):
    model = 'gbm'
    fair = 'reduction'
    seed = 42
    constraint = 'eo'
    num_iter = 10
    inputfile = 'adult'
    
    try:
        opts, args = getopt.getopt(argv,"hm:s:f:c:n:i:")
    except getopt.GetoptError:
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('Correct Usage:\n')
            print('python run_benchmark.py -m [model name] -f [fair method] -c [constraint] -n [num iter] -i [inputfile] -s [seed]')
            print('\n')
            print('Options for arguments:')
            print('[model name]: gbm, logit, rf (Default: gbm)')
            print('[fair method]: reduction, eqodds, roc (Default: reduction)')
            print('[constraint]: eo, sp, (Default: eo)')
            print('[num iter]: Any positive integer (Default: 10) ')
            print('[inputfile]: adult, adult_FATO, compas, compas_modified ...  (Default: compas_modified)')
            print('[seed]: Any integer (Default: 42)')
            print('\n')
            sys.exit()
        elif opt == "-m":
            model = arg
        elif opt  == "-s":
            seed = int(arg)
        elif opt == '-f':
            fair = arg
        elif opt == '-c':
            constraint = arg
        elif opt == '-n':
            num_iter = int(arg)
        elif opt == '-i':
            inputfile = arg
        
        
    if inputfile == 'adult':
        df = load_data('adult', modified = True, perturbed = False)
        privileged_groups = [{'gender': 1}]
        unprivileged_groups = [{'gender': 0}]
        protected_attrs = ['gender']
        label_name = 'income'

    elif inputfile == 'compas':
        df = load_data('compas', modified = True, perturbed = False)
        privileged_groups = [{'race': 1}]
        unprivileged_groups = [{'race': 0}]
        protected_attrs = ['race']
        label_name = 'is_recid'
     
    elif inputfile == 'german_credit':
        with open('../data/german_credit/german_data_processed.pkl', 'rb') as file:
            df = pickle.load(file)
            df.drop('SY',inplace=True,axis=1) #drop the "SY" column from input

        privileged_groups = [{'age': 1}]
        unprivileged_groups = [{'age': 0}]
        protected_attrs = ['age']
        label_name = 'credit'  
    else: 
        print('Invalid Input Dataset Name')
        sys.exit(2)

    print('#### Data Loaded. ')
    

    #### Setting group attribute and label ####
    
    
    bm = Benchmark(df, privileged_groups, unprivileged_groups, protected_attrs,label_name)


    #### Run benchmarks ####
    if fair == 'reduction':
        

        # adult
        eps_list = [ 0.001, 0.01, 0.2, 0.5, 1, 2, 5, 10,15, 20, 25, 30] 
        
        # compas
        #eps_list = [ 0.001, 2, 5, 10, 15, 20, 25, 30, 35, 40, 50] 
        
        # German credits
        #eps_list = [ 20,50,80,95]

        # add bigger epsilons to extend the reduction line
       

        if constraint == 'sp':
            # add inputfile as input to reduction()
            results = bm.reduction(model, num_iter, seed, params=eps_list, inputfile = inputfile, constraint='DemographicParity')
        elif constraint == 'eo':
            results = bm.reduction(model, num_iter, seed, params=eps_list, inputfile = inputfile, constraint='EqualizedOdds')
        
    elif fair == 'eqodds':
        results = bm.eqodds(model, num_iter, seed)
        constraint = ''
    
    elif fair == 'caleqodds': 
        results = bm.eqodds(model, num_iter, seed, calibrated=True, constraint=constraint)
        constraint = ''
        
    elif fair == 'roc':
        eps_list = [0.005, 0.0075, 0.01, 0.0125] # Metric ub and lb values for roc method #
        if constraint == 'sp':
            results = bm.roc(model, num_iter, seed, params=eps_list, constraint='DemographicParity')
        elif constraint == 'eo':
            results = bm.roc(model, num_iter, seed, params=eps_list, constraint='EqualizedOdds')
        
    elif fair == 'leveraging':
        _, results, _ = leveraging_approach(df, protected_attrs, label_name, use_protected=True, model = model, num_iter = num_iter, rand_seed =seed)
        
    elif fair == 'original':
        results = bm.original(model, num_iter, seed)
        constraint = ''
        
    else:
        print('Undefined method')
        sys.exit(2)


    result_path = './results/'
    filename = fair+'_'+model+'_s'+str(seed)+'_' + constraint+'.pkl'
    
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    with open(result_path+filename, 'wb+') as f: 
        pickle.dump(results, f)


if __name__ == "__main__":
    main(sys.argv[1:])
