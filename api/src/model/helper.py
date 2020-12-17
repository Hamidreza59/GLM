#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: hamidrezaouni
"""

import pandas as pd
from dataUtil import PrepareData
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt


class ModelHelper():
    def __init__(self):
        self.prepareData = PrepareData()
        print("__instance created")

    def preprocessing(self, df):
        """
        preprocessing data 
    
        Parameters
        ----------
        dateFame
    
        Returns
        -------
        dataFrame
        """
        getCategiricals = ['x5', 'x31', 'x81', 'x82']                    # list of catorical feature in dataset
        df['x12'] = self.prepareData.clean_x12(df['x12'])                # cleaning column x12
        df['x63'] = self.prepareData.clean_x63(df['x63'])                # cleaning column x63   
        for col, col_type in df.dtypes.iteritems():
            if col_type != 'object':
               df[col].fillna(df[col].mean(), inplace=True)              # put mean of feature into empty values
               df[col] = self.prepareData.normalize(df[col])             # Normalize each numeric column 
        df = pd.get_dummies(df, columns=getCategiricals, dummy_na=True)  # one hot encoding of categorical features in dataset
        return df


    def split(self, df, split_size, random_state):
        """
        split data in train and dev sets
    
        Parameters
        ----------
        df: dataFrame
        split_size: number to split between 0 and one
        random_state: int for random_state
        Returns
        -------
        trainset
        devset
        """
        return train_test_split(df, test_size= split_size, random_state=random_state)

    def feature_selection(self, X, Y, selectNum):
        """
        select best features modal based on correlation between features
    
        Parameters
        ----------
        X: dependent data
        Y: Response data
        selectNum: number of best feature to selct
        Returns
        -------
        list of seelect number of best features with highest coef_squared
        """
        exploratory_Linear = LogisticRegression(penalty='l1', fit_intercept=False, solver='liblinear')
        exploratory_Linear.fit(X, Y)
        results = pd.DataFrame(X.columns).rename(columns={0:'name'})
        results['coefs'] = exploratory_Linear.coef_[0]
        results['coefs_squared'] = results['coefs']**2
        
        return results.nlargest(selectNum,'coefs_squared')['name'].to_list()
    

    def correlation(self, df, showPlot):
        """
        find the correlation between variables and return the serise of strong 
        correlations (bigger than 0.5 and less than -0.5)
    
        Parameters
        ----------
        df: dataFrame
        showPlot: boolean to show heatmap of correlation matrix
    
        Returns
        -------
       the seriese of strong correlation
    
        """
        corr = df.corr()
        if showPlot:
            sns.heatmap(corr, annot = True)
            plt.title("Correlation matrix of data")
            plt.xlabel("features")
            plt.ylabel("features")
            plt.show()
        corr_pairs = corr.unstack()    
        sorted_pairs = corr_pairs.sort_values(kind="quicksort")
        strong_corr = sorted_pairs[abs(sorted_pairs) > 0.5]
        return strong_corr
    
    
    
    