# code for data pre-processing. Returns a dataframe with all columns including 'SY'.

import pandas as pd
import numpy as np
from sklearn import metrics as sm
import sys
import random 

rand_seed = 42

from sklearn.preprocessing import MinMaxScaler
#%% method for loading different datasets
def load_data(name='adult', perturbed = False, alpha = [0.1, 0.5]):
    '''
    If perturbed == True, then use mode imputation to perturb the majority and minority groups 
    with percentages indicated by parameter alpha. 
    
    eg. alpha = [0.1, 0.5] means each entry in the majority group has 0.1 probability of mode imputation,
    and each entry in the minority group has 0.5 probability of mode imputation.
    '''
    # Processing for UCI-ADULT
    if name == 'adult':
        file = '../data/UCI-Adult/adult.data'
        fileTest = '../data/UCI-Adult/adult.test'
        
        df = pd.read_csv(file, header=None,sep=',\s+',engine='python')
        dfTest = pd.read_csv(fileTest,header=None,skiprows=1,sep=',\s+',engine='python') 
        
        columnNames = ["age", "workclass", "fnlwgt", "education", "education-num",
    "marital-status", "occupation", "relationship", "race", "gender",
    "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"]
        
        df.columns = columnNames
        dfTest.columns = columnNames
        
        df = df.append(dfTest)
        
        # drop columns that won't be used
        dropCol = ["fnlwgt","workclass","occupation"]
        df.drop(dropCol,inplace=True,axis=1)
        
        # keep only entries marked as ``White'' or ``Black''
        ix = df['race'].isin(['White','Black'])
        df = df.loc[ix,:]
        
        # binarize race
        # Black = 0; White = 1
        df.loc[:,'race'] = df['race'].apply(lambda x: 1 if x=='White' else 0)
        
        # binarize gender
        # Female = 0; Male = 1
        df.loc[:,'gender'] = df['gender'].apply(lambda x: 1 if x=='Male' else 0)
        
        # binarize income
        # '>50k' = 1; '<=50k' = 0
        df.loc[:,'income'] = df['income'].apply(lambda x: 1 if x[0]=='>' else 0)


        # drop "education" and native-country (education already encoded in education-num)
        features_to_drop = ["education","native-country"]
        df.drop(features_to_drop,inplace=True,axis=1)

       
        # create one-hot encoding
        categorical_features = list(set(df)-set(df._get_numeric_data().columns))
        df = pd.concat([df,pd.get_dummies(df[categorical_features])],axis=1,sort=False)
        df.drop(categorical_features,inplace=True,axis=1)
        df.dropna()

        df = df[['gender', 'hours-per-week', 'education-num', 'age', 'marital-status_Married-civ-spouse', 
                 'relationship_Husband','relationship_Wife', 'income']]
        # reset index
        #df = df.reset_index()
        
        def perturb_row(row, column, alpha):

            # dictionary with all the modes
            mode = {
              'hours-per-week':40,
              'education-num':9,
              'age':36,
              'marital-status_Married-civ-spouse':0,
              'relationship_Husband':0,
              'relationship_Wife':0
          }

            majority_rate, minority_rate = alpha
            random_num = np.random.rand()

            if row['gender'] == 0: 
                if random_num <= minority_rate:
                    return mode[column]
                else:
                    return row[column]
            else:
                if random_num <= majority_rate:
                    return mode[column]    
                else:
                    return row[column]      

        if perturbed == True:
            for column in ['hours-per-week', 'education-num','age',  'marital-status_Married-civ-spouse','relationship_Husband','relationship_Wife']:
                df[column] = df.apply(lambda x: perturb_row(x, column, alpha), axis=1)
            

        def get_SY_column(row):
            if row['gender'] == 0 and row['income'] == 0:
                return 0
            elif row['gender'] == 0 and row['income'] == 1:
                return 1
            elif row['gender'] == 1 and row['income'] == 0:
                return 2
            else:
                return 3

        def get_discrete_age(row):
            '''
            discretize the age column into 12 buckets
            '''
            row_age = row['age'] 
            if row_age < 20:
                return 0
            elif row_age < 25:
                return 1
            elif row_age < 30:
                return 2
            elif row_age < 35:
                return 3
            elif row_age < 40:
                return 4
            elif row_age < 45:
                return 5
            elif row_age < 50:
                return 6
            elif row_age < 55:
                return 7
            elif row_age < 60:
                return 8
            elif row_age < 65:
                return 9
            elif row_age < 70:
                return 10
            else:
                return 11
          
            
        def get_discrete_hours(row):
            '''
            discretize hours per week column into 14 intervals.
            '''
            row_hours = row['hours-per-week']
            if row_hours < 10:
                return 0
            elif row_hours < 15:
                return 1
            elif row_hours < 20:
                return 2
            elif row_hours < 25:
                return 3
            elif row_hours < 30:
                return 4
            elif row_hours < 35:
                return 5
            elif row_hours < 40:
                return 6
            elif row_hours < 45:
                return 7
            elif row_hours < 50:
                return 8
            elif row_hours < 55:
                return 9
            elif row_hours < 60:
                return 10
            elif row_hours < 65:
                return 11
            elif row_hours < 70:
                return 12
            else:
                return 13    
            

        # discretize age, education-num, hours-per-week
        df['age_discrete'] = df.apply(lambda x: get_discrete_age(x), axis=1)
        df['hours_discrete'] = df.apply(lambda x: get_discrete_hours(x), axis=1)

        # without any processing
        df['edu_discrete'] = df['education-num']
        

        # add a joint (s,y) column
        df['SY'] = df.apply(lambda x: get_SY_column(x), axis=1)
        
    
    # Processing for COMPAS
    if name == 'compas':
        file = '../data/COMPAS/compas-scores-two-years.csv'
        df = pd.read_csv(file,index_col=0)
        
        # select features for analysis
        df = df[['age', 'c_charge_degree', 'race',  'sex', 'priors_count', 
                    'days_b_screening_arrest',  'is_recid',  'c_jail_in', 'c_jail_out']]
        
        # drop missing/bad features (following ProPublica's analysis)
        # ix is the index of variables we want to keep.

        # Remove entries with inconsistent arrest information.
        ix = df['days_b_screening_arrest'] <= 30
        ix = (df['days_b_screening_arrest'] >= -30) & ix

        # remove entries entries where compas case could not be found.
        ix = (df['is_recid'] != -1) & ix

        # remove traffic offenses.
        ix = (df['c_charge_degree'] != "O") & ix


        # trim dataset
        df = df.loc[ix,:]

        # create new attribute "length of stay" with total jail time.
        df['length_of_stay'] = (pd.to_datetime(df['c_jail_out'])-pd.to_datetime(df['c_jail_in'])).apply(lambda x: x.days)
        
        
        # drop 'c_jail_in' and 'c_jail_out'
        # drop columns that won't be used
        dropCol = ['c_jail_in', 'c_jail_out','days_b_screening_arrest']
        df.drop(dropCol,inplace=True,axis=1)
        
        # keep only African-American and Caucasian
        df = df.loc[df['race'].isin(['African-American','Caucasian']),:]
        
        # binarize race 
        # African-American: 0, Caucasian: 1
        df.loc[:,'race'] = df['race'].apply(lambda x: 1 if x=='Caucasian' else 0)
        
        # binarize gender
        # Female: 1, Male: 0
        df.loc[:,'sex'] = df['sex'].apply(lambda x: 1 if x=='Male' else 0)
        
        # rename columns 'sex' to 'gender'
        df.rename(index=str, columns={"sex": "gender"},inplace=True)
        
        # binarize degree charged
        # Misd. = -1, Felony = 1
        df.loc[:,'c_charge_degree'] = df['c_charge_degree'].apply(lambda x: 1 if x=='F' else -1)

        # reset index 
        df.reset_index(inplace=True,drop=True)

        def perturb_row(row, column, alpha):

            # dictionary with all the modes
            mode = {
              'age': 24,
              'gender':1,
              'c_charge_degree':1,
              'priors_count':0,
              'length_of_stay':0
          }
            majority_rate, minority_rate = alpha
            random_num = np.random.rand()
            if row['race'] == 0:
                if random_num <= minority_rate:
                    return mode[column]
                else:
                    return row[column]
            else:
                if random_num <= majority_rate:
                    return mode[column]    
                else:
                    return row[column]      

        if perturbed == True:
          for column in ['age', 'gender', 'c_charge_degree','priors_count','length_of_stay']:
            df[column] = df.apply(lambda x: perturb_row(x, column, alpha), axis=1)
          
                  
        # add a SY column 
        def get_SY_column(row):
            if row['race'] == 0 and row['is_recid'] == 0:
                return 0
            elif row['race'] == 0 and row['is_recid'] == 1:
                return 1
            elif row['race'] == 1 and row['is_recid'] == 0:
                return 2
            else:
                return 3


        # discretize the length_of_stay column 
        def get_discrete_stay(row):
            '''
            discretize length_of_stay column by month (30 days).
            '''
            row_length = row['length_of_stay']
            if row_length <= 0:
                output =  0
            else:
                output =  row_length // 30 + 1
            return output 

       

        # discretize age column
        def get_discrete_age(row):
            '''
            discretize the age column into 12 buckets
            '''
            row_age = row['age'] 
            if row_age < 20:
                return 0
            elif row_age < 25:
                return 1
            elif row_age < 30:
                return 2
            elif row_age < 35:
                return 3
            elif row_age < 40:
                return 4
            elif row_age < 45:
                return 5
            elif row_age < 50:
                return 6
            elif row_age < 55:
                return 7
            elif row_age < 60:
                return 8
            elif row_age < 65:
                return 9
            elif row_age < 70:
                return 10
            else:
                return 11
        
        df['SY'] = df.apply(lambda x: get_SY_column(x), axis=1)
        df['stay_discrete'] = df.apply(lambda x: get_discrete_stay(x), axis=1)
        df['age_discrete'] = df.apply(lambda x: get_discrete_age(x), axis=1)


    return df



