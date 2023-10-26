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


from IPython.display import Markdown, display
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression 
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder


from aif360.datasets import BinaryLabelDataset
from aif360.datasets import AdultDataset, GermanDataset, CompasDataset, StandardDataset
from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric
from aif360.metrics import ClassificationMetric
from aif360.metrics.utils import compute_boolean_conditioning_vector

from aif360.algorithms.inprocessing.exponentiated_gradient_reduction import ExponentiatedGradientReduction
from aif360.algorithms.postprocessing.calibrated_eq_odds_postprocessing import CalibratedEqOddsPostprocessing
from aif360.algorithms.postprocessing.eq_odds_postprocessing import EqOddsPostprocessing
from aif360.algorithms.postprocessing.reject_option_classification import RejectOptionClassification



class Benchmark:
    
    def __init__(self, df,privileged_groups, unprivileged_groups, protected_attrs,label_name):
        self._df = df
        self._privileged_groups = privileged_groups
        self._unprivileged_groups = unprivileged_groups
        self._protected_attrs = protected_attrs
        self._label_name = label_name
        

    def get_idx_wo_protected(self, feature_names, protected_attrs):
        idx_wo_protected = set(range(len(feature_names)))
        protected_attr_idx = [feature_names.index(x) for x in protected_attrs]
        idx_wo_protected = list(idx_wo_protected-set(protected_attr_idx))
        return idx_wo_protected
    

    def make_result_dict(self, stat_names, other_names): 
        results = {}
        for name in stat_names: 
            results[name+'_mean'] = [] 
            results[name+'_std'] = [] 
        for name in other_names:
            results[name] = []
            
        return results
    
    
    def update_subresults(self, results, cm_transf_test, metric_names):
        def abseo_metric(cm_transf_test):
            '''
            Take the max of absolute difference in FPR and TPR
            '''
            return max(np.abs(cm_transf_test.difference(cm_transf_test.false_positive_rate)), np.abs(cm_transf_test.difference(cm_transf_test.true_positive_rate)))

        for name in metric_names: 
            if name == 'sp':
                eval_metric =  np.abs(cm_transf_test.statistical_parity_difference())
            elif name == 'avgeo': 
                eval_metric =  np.abs(cm_transf_test.average_odds_difference())
            elif name == 'abseo':
                eval_metric =  abseo_metric(cm_transf_test)
            elif name == 'maxeo': 
                eval_metric =  max(cm_transf_test.false_negative_rate_difference(), 
                                cm_transf_test.false_positive_rate_difference())
            elif name == 'acc': 
                eval_metric = cm_transf_test.accuracy()
            
            results[name].append(eval_metric)
            
    def print_subresults(self, results, metric_names): 
        for name in metric_names:
            line = ''
            if name == 'avgeo': 
                line += 'Average odds difference:'
            elif name == 'sp': 
                line += 'Statistical parity difference:'
            elif name == 'abseo': 
                line += 'Max equalized odds:'
            elif name == 'brier':
                line += 'Overall Test Brier Score:'
            elif name == 'acc':
                line += 'Overall Test Accuracy:'
            elif name == 'time': 
                line += 'Time per iteration:'
            
            line += str(results[name][-1])    
            print(line)
            
        print('----------------------------------------------------------')

       
    
    def reduction(self, model_name='gbm', num_iter=10, rand_seed=42, constraint='EqualizedOdds', inputfile = 'adult', params=[0.1]): 
        '''
        define the reduction approach. Inputfile variable is the dataset name, such as 'adult', 'compas' etc.
        '''
        metric_names = ['sp', 'avgeo', 'abseo', 'maxeo', 'acc', 'brier', 'time']
        other_names = ['eps']
        results = self.make_result_dict(metric_names, other_names)
        
        for eps in params: 
            
            print("## Epsilon ="+str(eps))
            sub_results = self.make_result_dict([], metric_names)

            for seed in range (num_iter): 
                print('Iteration #', seed)
                start = time.time()

                # instead of train/test split, we sample with replacement 
                dataset_orig_train = self._df
                dataset_orig_test = (self._df).sample(frac=0.3, replace=True, random_state=seed)
              
                
                ### Converting to AIF360 StandardDataset objects ###
                dataset_orig_train = StandardDataset(dataset_orig_train, label_name=self._label_name, favorable_classes=[1],
                                                    protected_attribute_names=self._protected_attrs, privileged_classes=[[1]])
                dataset_orig_test = StandardDataset(dataset_orig_test, label_name=self._label_name, favorable_classes=[1],
                                                    protected_attribute_names=self._protected_attrs, privileged_classes=[[1]])
                
        
                X_train, y_train = dataset_orig_train.features, dataset_orig_train.labels.ravel()
                X_test, y_test = dataset_orig_test.features, dataset_orig_test.labels.ravel()


                if model_name == 'gbm': 
                    model = GradientBoostingClassifier(random_state=rand_seed)
                elif model_name == 'logit': 
                    model = LogisticRegression(random_state=rand_seed)
                elif model_name == 'rf':
                    # TODO: adult
                    model = RandomForestClassifier(random_state=rand_seed, n_estimators=15, min_samples_leaf=3, criterion = 'log_loss', bootstrap = False) 
                    
                    # TODO: compas
                     #model =  RandomForestClassifier(random_state=rand_seed, n_estimators=17)
                    
                else: 
                    print('Error: Undefined Model')
                    return 
        
                
                ### Train A Classifier Using The Reductions Approach ### 
                
                np.random.seed(rand_seed)
                exp_grad_red = ExponentiatedGradientReduction(estimator=model, 
                                                        constraints=constraint,
                                                        drop_prot_attr=False, eps=eps)                

                exp_grad_red.fit(dataset_orig_train)
                
               
                exp_grad_red_pred = exp_grad_red.predict(dataset_orig_test)

                cm_transf_test = ClassificationMetric(dataset_orig_test, exp_grad_red_pred,
                                                unprivileged_groups=self._unprivileged_groups,
                                                privileged_groups=self._privileged_groups)

                
                self.update_subresults(sub_results, cm_transf_test, metric_names)
                
                brier = brier_score_loss(y_test, np.clip(exp_grad_red_pred.scores, 0,1))
                sub_results['brier'].append(brier)
                
                end = time.time()
                sub_results['time'].append(end-start)
                
                
                if constraint == 'EqualizedOdds':
                    self.print_subresults(sub_results, ['abseo', 'acc', 'time'])
                else: 
                    self.print_subresults(sub_results, ['sp', 'acc', 'time'])
                
            for s in metric_names:
                results[s+'_mean'].append(np.mean(sub_results[s]))
                results[s+'_std'].append(np.std(sub_results[s]))
                
            results['eps'] = params

        return results


    def eqodds(self, model_name='gbm', num_iter=10, rand_seed=42, calibrated=False, constraint='weighted'): 
        metric_names = ['sp', 'avgeo', 'abseo', 'maxeo', 'acc', 'brier', 'time']
        results = self.make_result_dict(metric_names, [])
        sub_results = self.make_result_dict([], metric_names)


        for seed in range (num_iter): 
            print('Iteration #', seed)
            start = time.time()
            
            dataset_orig_train = self._df
            dataset_orig_test = (self._df).sample(frac=0.3, replace=True, random_state=seed)
            dataset_orig_valid = (self._df).sample(frac=0.15, replace=True, random_state=seed)

            ### Converting to AIF360 StandardDataset objects ###
            dataset_orig_train = StandardDataset(dataset_orig_train, label_name=self._label_name, favorable_classes=[1],
                                                 protected_attribute_names=self._protected_attrs, privileged_classes=[[1]])
            dataset_orig_valid = StandardDataset(dataset_orig_valid, label_name=self._label_name, favorable_classes=[1],
                                                protected_attribute_names=self._protected_attrs, privileged_classes=[[1]])
            dataset_orig_test = StandardDataset(dataset_orig_test, label_name=self._label_name, favorable_classes=[1],
                                                protected_attribute_names=self._protected_attrs, privileged_classes=[[1]])

            # Placeholder for predicted and transformed datasets
            dataset_orig_train_pred = dataset_orig_train.copy(deepcopy=True)
            dataset_orig_valid_pred = dataset_orig_valid.copy(deepcopy=True)
            dataset_orig_test_pred = dataset_orig_test.copy(deepcopy=True)
            
            
        

            #idx_wo_protected = self.get_idx_wo_protected(dataset_orig_train.feature_names, self._protected_attrs)

            X_train, y_train = dataset_orig_train.features, dataset_orig_train.labels.ravel()
            X_test, y_test = dataset_orig_test.features, dataset_orig_test.labels.ravel()


            if model_name == 'gbm': 
                model = GradientBoostingClassifier(random_state=rand_seed)
            elif model_name == 'logit': 
                model = LogisticRegression(random_state=rand_seed)
            elif model_name == 'rf':

                # adult
                model = RandomForestClassifier(random_state=rand_seed, n_estimators=15, min_samples_leaf=3, criterion = 'log_loss', bootstrap = False) 

                # COMPAS
                # model =  RandomForestClassifier(random_state=rand_seed, n_estimators=17)
             
            else: 
                print('Error: Undefined Model')
                return 


            ### Train Original Classifier ### 
            model.fit(X_train, y_train, sample_weight=dataset_orig_train.instance_weights)


            fav_idx = np.where(model.classes_ == dataset_orig_train.favorable_label)[0][0]
            y_train_pred_prob = model.predict_proba(X_train)[:,fav_idx]

            # Prediction probs for validation and testing data
            X_valid = dataset_orig_valid.features
            y_valid_pred_prob = model.predict_proba(X_valid)[:,fav_idx]

            X_test = dataset_orig_test.features
            y_test_pred_prob = model.predict_proba(X_test)[:,fav_idx]

            class_thresh = 0.5
            dataset_orig_train_pred.scores = y_train_pred_prob.reshape(-1,1)
            dataset_orig_valid_pred.scores = y_valid_pred_prob.reshape(-1,1)
            dataset_orig_test_pred.scores = y_test_pred_prob.reshape(-1,1)

            y_train_pred = np.zeros_like(dataset_orig_train_pred.labels)
            y_train_pred[y_train_pred_prob >= class_thresh] = dataset_orig_train_pred.favorable_label
            y_train_pred[~(y_train_pred_prob >= class_thresh)] = dataset_orig_train_pred.unfavorable_label
            dataset_orig_train_pred.labels = y_train_pred

            y_valid_pred = np.zeros_like(dataset_orig_valid_pred.labels)
            y_valid_pred[y_valid_pred_prob >= class_thresh] = dataset_orig_valid_pred.favorable_label
            y_valid_pred[~(y_valid_pred_prob >= class_thresh)] = dataset_orig_valid_pred.unfavorable_label
            dataset_orig_valid_pred.labels = y_valid_pred

            y_test_pred = np.zeros_like(dataset_orig_test_pred.labels)
            y_test_pred[y_test_pred_prob >= class_thresh] = dataset_orig_test_pred.favorable_label
            y_test_pred[~(y_test_pred_prob >= class_thresh)] = dataset_orig_test_pred.unfavorable_label
            dataset_orig_test_pred.labels = y_test_pred

            if calibrated: 
                cpp = CalibratedEqOddsPostprocessing(privileged_groups = self._privileged_groups,
                                                     unprivileged_groups = self._unprivileged_groups,seed=rand_seed,
                                                     cost_constraint='fnr'
                                                    ) #COMPAS: cost_constraint='weighted'; Adult: cost_constraint='fnr'
            else: 
                cpp = EqOddsPostprocessing(privileged_groups = self._privileged_groups,
                                           unprivileged_groups = self._unprivileged_groups,seed=rand_seed)
                
            cpp = cpp.fit(dataset_orig_valid, dataset_orig_valid_pred)

            dataset_transf_test_pred = cpp.predict(dataset_orig_test_pred)

            cm_transf_test = ClassificationMetric(dataset_orig_test, dataset_transf_test_pred,
                            unprivileged_groups=self._unprivileged_groups,
                            privileged_groups=self._privileged_groups)

            self.update_subresults(sub_results, cm_transf_test, metric_names)
                
            brier = brier_score_loss(y_test, np.clip(dataset_transf_test_pred.scores, 0,1))
            sub_results['brier'].append(brier)
            
            end = time.time()
            sub_results['time'].append(end-start)
            
            
            self.print_subresults(sub_results, ['abseo', 'acc', 'time'])
            
        for s in metric_names:
            results[s+'_mean'].append(np.mean(sub_results[s]))
            results[s+'_std'].append(np.std(sub_results[s]))

        return results
            


    
    def roc(self, model_name='gbm', num_iter=10, rand_seed=42, constraint='EqualizedOdds', params=[0.1]): 
        
        metric_names = ['sp', 'avgeo', 'abseo', 'maxeo', 'acc', 'brier', 'time']
        other_names = ['eps']
        results = self.make_result_dict(metric_names, other_names)
        
        for eps in params: 
            
            print("## Epsilon ="+str(eps))
            sub_results = self.make_result_dict([], metric_names)
            
            for seed in range (num_iter): 
                print('Iteration #', seed)
                
                start = time.time()
                

                dataset_orig_train = self._df
                dataset_orig_test = (self._df).sample(frac=0.3, replace=True, random_state=seed)
                dataset_orig_valid = (self._df).sample(frac=0.15, replace=True, random_state=seed)

                ### Converting to AIF360 StandardDataset objects ###
                dataset_orig_train = StandardDataset(dataset_orig_train, label_name=self._label_name, favorable_classes=[1],
                                                    protected_attribute_names=self._protected_attrs, privileged_classes=[[1]])
                dataset_orig_valid = StandardDataset(dataset_orig_valid, label_name=self._label_name, favorable_classes=[1],
                                                    protected_attribute_names=self._protected_attrs, privileged_classes=[[1]])
                dataset_orig_test = StandardDataset(dataset_orig_test, label_name=self._label_name, favorable_classes=[1],
                                                    protected_attribute_names=self._protected_attrs, privileged_classes=[[1]])
     
                #idx_wo_protected = self.get_idx_wo_protected(dataset_orig_train.feature_names, self._protected_attrs)

                # remove protected features
                X_train, y_train = dataset_orig_train.features, dataset_orig_train.labels.ravel()
                X_valid, y_valid = dataset_orig_valid.features, dataset_orig_valid.labels.ravel()
                X_test, y_test = dataset_orig_test.features, dataset_orig_test.labels.ravel()
                
                if model_name == 'gbm': 
                    model = GradientBoostingClassifier(random_state=rand_seed)
                elif model_name == 'logit': 
                    model = LogisticRegression(random_state=rand_seed)
                elif model_name == 'rf':
                    # Adult
                    model = RandomForestClassifier(random_state=rand_seed, n_estimators=15, min_samples_leaf=3, criterion = 'log_loss', bootstrap = False) 

                    # COMPAS
                    # model =  RandomForestClassifier(random_state=rand_seed, n_estimators=17)
                else: 
                    print('Error: Undefined Model')
                    return 

                model.fit(X_train, y_train)
                y_train_pred = model.predict(X_train)

                # positive class index
                pos_ind = np.where(model.classes_ == dataset_orig_train.favorable_label)[0][0]

                dataset_orig_train_pred = dataset_orig_train.copy(deepcopy=True)
                dataset_orig_train_pred.labels = y_train_pred

                dataset_orig_valid_pred = dataset_orig_valid.copy(deepcopy=True)
                dataset_orig_valid_pred.scores = model.predict_proba(X_valid)[:,pos_ind].reshape(-1,1)

                dataset_orig_test_pred = dataset_orig_test.copy(deepcopy=True)
                dataset_orig_test_pred.scores = model.predict_proba(X_test)[:,pos_ind].reshape(-1,1)

                if constraint == 'EqualizedOdds':
                    metric_name = "Average odds difference"
                elif constraint == 'DemographicParity':
                    metric_name = "Statistical parity difference"
                else:
                    print('Error: Unknown constraint')
                    return 
                
                ROC = RejectOptionClassification(unprivileged_groups=self._unprivileged_groups, 
                                                privileged_groups=self._privileged_groups, 
                                                low_class_thresh=0.1, high_class_thresh=0.9,
                                                num_class_thresh=80, num_ROC_margin=40,
                                                metric_name=metric_name,
                                                metric_ub=eps, metric_lb=-eps)
                ROC = ROC.fit(dataset_orig_valid, dataset_orig_valid_pred)
                
                print("Optimal classification threshold (with fairness constraints) = %.4f" % ROC.classification_threshold)
                print("Optimal ROC margin = %.4f" % ROC.ROC_margin)
                
                dataset_transf_test_pred = ROC.predict(dataset_orig_test_pred)
                
                cm_transf_test = ClassificationMetric(dataset_orig_test, dataset_transf_test_pred,
                                unprivileged_groups=self._unprivileged_groups,
                                privileged_groups=self._privileged_groups)
                
                
                self.update_subresults(sub_results, cm_transf_test, metric_names)
                
                brier = brier_score_loss(y_test, np.clip(dataset_transf_test_pred.scores, 0,1))
                sub_results['brier'].append(brier)
                
                end = time.time()
                sub_results['time'].append(end-start)
                
                
                if constraint == 'EqualizedOdds':
                    self.print_subresults(sub_results, ['abseo', 'acc', 'time'])
                else: 
                    self.print_subresults(sub_results, ['sp', 'acc', 'time'])
                
            for s in metric_names:
                results[s+'_mean'].append(np.mean(sub_results[s]))
                results[s+'_std'].append(np.std(sub_results[s]))
                
            results['eps'] = params

        return results

    def original(self, model_name='gbm', num_iter=10, rand_seed=42): 
        
        metric_names = ['sp', 'avgeo', 'abseo', 'maxeo', 'acc', 'brier', 'time']
        results = self.make_result_dict(metric_names, [])
        sub_results = self.make_result_dict([], metric_names)
        

        for seed in range (num_iter): 
            
            print('Iteration #', seed)
            start = time.time()
            

            dataset_orig_train = self._df
            dataset_orig_test = (self._df).sample(frac=0.3, replace=True, random_state=seed)

            ### Converting to AIF360 StandardDataset objects ###
            dataset_orig_train = StandardDataset(dataset_orig_train, label_name=self._label_name, favorable_classes=[1],
                                                protected_attribute_names=self._protected_attrs, privileged_classes=[[1]])
            dataset_orig_test = StandardDataset(dataset_orig_test, label_name=self._label_name, favorable_classes=[1],
                                                protected_attribute_names=self._protected_attrs, privileged_classes=[[1]])



            X_train, y_train = dataset_orig_train.features, dataset_orig_train.labels.ravel()
            X_test, y_test = dataset_orig_test.features, dataset_orig_test.labels.ravel()

            if model_name == 'gbm': 
                model = GradientBoostingClassifier(random_state=rand_seed)
            elif model_name == 'logit': 
                model = LogisticRegression(random_state=rand_seed)
            elif model_name == 'rf':
                # TODO: adult
                model = RandomForestClassifier(random_state=rand_seed, n_estimators=15, min_samples_leaf=3, criterion = 'log_loss', bootstrap = False)
                
                # TODO: compas
                # model =  RandomForestClassifier(random_state=rand_seed, n_estimators=17)
                
            else: 
                print('Error: Undefined Model')
                return 


            ### Train Original Classifier ### 
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)

            dataset_orig_test_pred = dataset_orig_test.copy(deepcopy=True)
            dataset_orig_test_pred.labels = y_pred

            cm_pred_test = ClassificationMetric(dataset_orig_test, dataset_orig_test_pred,
                                    unprivileged_groups=self._unprivileged_groups,
                                    privileged_groups=self._privileged_groups)

            self.update_subresults(sub_results, cm_pred_test, metric_names)
                
            brier = brier_score_loss(y_test, np.clip(dataset_orig_test_pred.scores, 0,1))
            sub_results['brier'].append(brier)
            
            end = time.time()
            sub_results['time'].append(end-start)
            
            
            self.print_subresults(sub_results, ['abseo', 'acc', 'time'])
            
        for s in metric_names:
            results[s+'_mean'].append(np.mean(sub_results[s]))
            results[s+'_std'].append(np.std(sub_results[s]))

        return results

