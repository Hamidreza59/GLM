#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: hamidrezaouni
"""

import pandas as pd
import statsmodels.api as sm
from helper import ModelHelper
from dataUtil import PrepareData
import pickle

class GLMModel():
    
   def __init__(self, X_data, Y_data):
       self.X_data = X_data 
       self.Y_data = Y_data
       
    
   def model(self):
        """
        best model to fit the data
        Parameters
        ----------
        X_data : dataFrame(dependent variable)
        Y_data : dataFrame(response)
        Returns
        -------
        retun object {fit, variables}
        """
        self.X_data = helper.preprocessing(self.X_data)                                           # cleanning datasets
        data =  pd.concat([self.X_data,self.Y_data], axis=1, sort=False).reset_index(drop=True)   # combine all togetther after cleaning
        if helper.correlation(data, False).size > 50:                                             # check if there is strong correlation between features
            variables = helper.feature_selection(self.X_data, self.Y_data, 25)                    # if there is strong correlation do feature selection
        else:
            variables = self.X_data.columns
        
        logit = sm.Logit(data['y'], data[variables])
        result = logit.fit()
        return {'fit': result,'variables': variables} 
    



def apply_model(path):
    """
    this function get the path to data and saved model 
    and feature selected for production
    Parameters
    ----------
    path : Path to train data

    Returns
    -------
    None.

    """
    dataFrame = pd.read_csv(path)
    Y_data = dataFrame['y']
    X_data = dataFrame.drop(columns=['y'])
        
    glmModel = GLMModel(X_data, Y_data)
    model = glmModel.model()
    
    # lb.dump(model['fit'], 'model.pkl')
    print("Model dumped!")
        
    # Load the model that you just saved
    pkl_model = "pickle_model.pkl"
    with open(pkl_model, 'wb') as file:
        pickle.dump(model['fit'], file)
        
    # Saving the data columns from training
    pkl_varable = "pickle_variable.pkl"
    with open(pkl_varable, 'wb') as file:
        pickle.dump(list(model['variables']), file)
    
    print("Models columns dumped!")
   

if __name__ == '__main__':
    prepareData = PrepareData()
    helper = ModelHelper()
    apply_model("/Users/hamidrezaouni/Downloads/exercise_26_train.csv")


    