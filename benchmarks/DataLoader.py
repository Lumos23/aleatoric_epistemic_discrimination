import pandas as pd
import numpy as np
from sklearn import metrics as sm
import sys
from sklearn.preprocessing import MinMaxScaler
import random

rand_seed = 42

#%% method for loading different datasets
def load_data(name='adult', modified = True, perturbed = False):
    
    #% Processing for UCI-ADULT
        
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
        
        # reset index
        df.reset_index(inplace=True,drop=True)

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
        
        def perturb_row(row, column, alpha = [0.1, 0.3]):
            '''
             for each data column in each row, there is a default 10% (30%) probability of 
             it being perturbed to the mode of the column if it belongs to the majority (minority) group.
            '''
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
                df[column] = df.apply(lambda x: perturb_row(x, column), axis=1)
       
                       
        if modified == True:
            '''
            using the selected and discretized variables as in FATO approach
            '''    
            # discretize age, education-num, hours-per-week
            df['age_discrete'] = df.apply(lambda x: get_discrete_age(x), axis=1)
            df['hours_discrete'] = df.apply(lambda x: get_discrete_hours(x), axis=1)
            df['edu_discrete'] = df['education-num']

            # if using the modified version, we only keep certain columns
            df = df[['gender', 'hours_discrete', 'edu_discrete', 'age_discrete', 'marital-status_Married-civ-spouse' , 'relationship_Husband','relationship_Wife', 'income']]
    

    #% Processing for COMPAS
    elif name == 'compas':
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
     
        def perturb_row(row, column, alpha = [0.1, 0.3]):
            '''
            for each data column in each row, there is a default 10% (30%) probability of 
            it being perturbed to the mode of the column if it belongs to the majority (minority) group.
            '''
            mode = {
                'age': 24,
                'gender': 1,
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
            discretize the age column into 3 buckets
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
        
        if perturbed == True:
          for column in ['age', 'c_charge_degree','priors_count','length_of_stay']:
            df[column] = df.apply(lambda x: perturb_row(x, column), axis=1)
          
        if modified == True:
            df['age_discrete'] = df.apply(lambda x: get_discrete_age(x), axis=1)
            df['stay_discrete'] = df.apply(lambda x: get_discrete_stay(x), axis=1)
            # if using the modified version, we only keep certain columns
            df = df[['age_discrete', 'c_charge_degree','race', 'gender', 'priors_count',
                    'stay_discrete','is_recid'] ]

    return df

#%% method for computing fitting models on dataset. Returns P_{Y,S|X}, P_{Y|X}, P_{S|X}, and P_{S,Y}
class clf:
    
    def __init__(self,df,model_dict,S=[],Y=[],X=[]):
        """Initialize the predictive model class.
        
        This function receives a dataframe df, a dictionary of scikitlearn model (must have fit/predict functionality).
        It also takes a list of variablenames for S, Y, and X.
        The variable names must be valid column names of df.
        """
        
        # train values -- converting to cateagorical
        
        # create categorical dictionary and labels for YS
        self.YSdict = self.create_dict(df,Y+S) 
        ys_train = [self.YSdict[tuple(x)] for x in df[Y+S].values]
        
        
        # create categorical dictionary for Y
        if len(Y) == 1:
             y_train = df[Y].values
        else: 
            self.Ydict = self.create_dict(df,Y)
            y_train = [self.Ydict[tuple(x)] for x in df[Y].values]
        
        # create categorical dictionary for S
        if len(S) == 1:
            s_train = df[S].values
        else:
            self.Sdict = self.create_dict(df,S) 
            s_train = [self.Sdict[tuple(x)] for x in df[S].values]
        
        # declare models --- add new models and change parameters here!
        self.mPys_x = model_dict['Pys_x']
        self.mPy_x  = model_dict['Py_x']
        self.mPs_x = model_dict['Ps_x']
        
        #%%% fit models
        # Pys_x
        self.mPys_x.fit(df[X],ys_train)
     
        ys_predict = self.mPys_x.predict_proba(df[X])
        print('Training acc Pys_x: ' + str(sm.accuracy_score(np.argmax(ys_predict,axis=1),ys_train)))
        
        # Py_x
        self.mPy_x.fit(df[X],y_train)
        
        y_predict = self.mPy_x.predict_proba(df[X])
        print('Training acc Py_x: ' + str(sm.accuracy_score(np.argmax(y_predict,axis=1),y_train)))
        
        # Ps_x
        self.mPs_x.fit(df[X],s_train)
        
        s_predict = self.mPs_x.predict_proba(df[X])
        print('Training acc Ps_x: ' + str(sm.accuracy_score(np.argmax(s_predict,axis=1),s_train)))
        
        
        
        # print training errors
        
        
        # compute marginals
        self.mPys = df.groupby(Y+S).size().unstack(S)/len(df)
        
    #helper function for creating categorical dict
    def create_dict(self,df,features):
        featureDict = df.groupby(features).size()
        featureDict[:] = range(len(featureDict))
        featureDict = featureDict.to_dict()
        return featureDict
        

        
        
        
    
