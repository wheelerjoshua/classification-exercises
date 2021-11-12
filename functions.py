from scipy import stats
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pydataset import data
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer


def plot_hist(df, train):
    '''
    Takes in df and train dataframes and plots hist of numeric values
    '''
    for column in df.columns[(df.dtypes == float) | (df.dtypes == int)]:
        plt.figure()
        train[column].hist()
        plt.title(column)
        
def plot_box(df, train):
    '''
    Takes in df and train dataframes and plots boxplots of numeric values
    '''
    for column in df.columns[(df.dtypes == float) | (df.dtypes == int)]:
        plt.figure()
        plt.boxplot(train[column])
        plt.title(column)