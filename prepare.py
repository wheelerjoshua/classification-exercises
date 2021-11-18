import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# import splitting and imputing functions
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

# turn off pink boxes for demo
import warnings
warnings.filterwarnings("ignore")

# import our own acquire module
import acquire

####### LESSON FUNCTIONS #######

# def clean_data(df):
#     '''
#     This function will clean the data...'''
#     df.drop_duplicates(inplace = True)
#     cols_to_drop = ['deck', 'embarked', 'class','age']
#     df = df.drop(columns = cols_to_drop)
#     df['embark_town'] = df.embark_town.fillna(value = 'Southampton')
#     dummy_df = pd.get_dummies(df[['sex', 'embark_town']], dummy_na = False, drop_first = [True, True])
#     df = pd.concat([df, dummy_df], axis = 1)
#     return df

# def split_data(df):
#     '''
#     Takes in a dataframe and return train, validate, test subset dataframes
#     '''
#     train, test = train_test_split(df, test_size = .2, random_state=123, stratify = df.survived)
#     train, validate = train_test_split(train, test_size = .3, random_state = 123, stratify = train.survived)
#     return train, validate, test

# def impute_mode(train, validate, test):
#     '''
#     Takes in train, validate, and test, and uses train to identify the best value to replace nulls in embark_town
#     Imputes that value into all three sets and returns all three sets.
#     '''
#     imputer = SimpleImputer(missing_values = np.nan, strategy = 'most_frequent')
#     train[['embark_town']] = imputer.fit_transform(train[['embark_town']])
#     validate[['embark_town']] = imputer.transform(validate[['embark_town']])
#     test[['embark_town']] = imputer.transform(test[['embark_town']])
#     return train, validate, test

# def prep_titanic_data(df):
#     '''
#     Cleans data and preps train, validate, and test df's.
#     '''
#     df = clean_data(df)
#     train, validate, test = split_data(df)
#     return train, validate, test



####### EXERCISE FUNCTIONS #######

##### IRIS
def prep_iris(df):
    '''
    Takes in Iris dataframe and returns a prepared version of the dataframe
    '''
    df.drop(columns = 'species_id', inplace = True)
    df.rename(columns = {'species_name':'species'}, inplace = True)
    dummy_df = pd.get_dummies(df[['species']], dummy_na = False, drop_first = [True, True])
    df = pd.concat([df, dummy_df], axis = 1)
    df.rename(columns = {'species_versicolor':'versicolor', 'species_virginica':'virginica'}, inplace = True)

    return df

##### TITANIC
def prep_titanic(df):
    '''
    Takes in titanic dataframe and returns prepared version of the dataframe
    '''
    df.drop_duplicates(inplace = True)
    dummy_df = pd.get_dummies(df[['sex','embark_town']], dummy_na = False, drop_first = [True, True])
    df.drop(columns = ['sex', 'embark_town', 'age', 'embarked', 'deck','class'], inplace=True)

    df = pd.concat([df, dummy_df], axis = 1)
    return df

##### TELCO
def prep_telco(df):
    '''
    Takes in titanic dataframe and returns prepared version of the dataframe
    '''
    # Drop duplicates
    df.drop_duplicates(inplace = True)
    ####### Change total_charges to float #######
    df.total_charges = df.total_charges.str.strip()
    df = df[df.total_charges != ""]
    df.total_charges = df.total_charges.astype(float)
    #############################################
    dummy_df = pd.get_dummies(df[['gender','partner','dependents','multiple_lines', 'streaming_tv','streaming_movies','paperless_billing','churn',]], dummy_na = False, drop_first = [True, True])
    df = pd.concat([df, dummy_df], axis = 1)
    # Drop unecessary columns
    df.drop(columns = ['streaming_tv_No internet service','streaming_movies_No internet service','customer_id', 'online_security', 'online_backup', 'device_protection', 'tech_support', 'internet_service_type', 'payment_type','contract_type'], inplace = True)


    return df


##### SPLIT DATA
def split_iris(df):
    '''
    Takes in a dataframe and stratify variable, returns train, validate, test subset dataframes. 
    '''
    train, test = train_test_split(df, test_size = .2, stratify = df.species)
    train, validate = train_test_split(train, test_size = .3, stratify = train.species)
    return train, validate, test

def split_titanic(df):
    '''
    Takes in a dataframe and stratify variable, returns train, validate, test subset dataframes. 
    '''
    train, test = train_test_split(df, test_size = .2, stratify = df.survived)
    train, validate = train_test_split(train, test_size = .3, stratify = train.survived)
    return train, validate, test

def split_telco(df):
    '''
    Takes in a dataframe and stratify variable, returns train, validate, test subset dataframes. 
    '''
    train, test = train_test_split(df, test_size = .2, stratify = df.churn)
    train, validate = train_test_split(train, test_size = .3, stratify = train.churn)
    return train, validate, test