#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: hamidrezaouni
"""

import pandas as pd
import numpy as np

class PrepareData:
    def __init__(self):
        print("__instance created")
    
    def clean_x12(self, col):
        """
        cleaning column x12   
    
        Parameters
        ----------
        col 
    
        Returns
        -------
        col datafrmae
        """
        col = col.str.replace('$','')
        col = col.str.replace(',','')
        col = col.str.replace(')','')
        col = col.str.replace('(','')
        col = col.astype(float)        
        return col

    def clean_x63(self, col):
        """
        cleaning column x63  
    
        Parameters
        ----------
        col 
    
        Returns
        -------
        col datafrmae
        """
        col = col.str.replace('%','')
        col = col.astype(float)  
        
        return col
    
    def normalize(self, col):
        """
        normalize column data
    
        Parameters
        ----------
        column fo dataFrame
    
        Returns
        -------
        column of dataFrame
        """
        return (col - col.mean())/col.std() 
 
    
    def dataPreprocessing(self, df):
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
        df['x12'] = self.clean_x12(df['x12'])                            # cleaning column x12
        df['x63'] = self.clean_x63(df['x63'])                            # cleaning column x63   
        df = pd.get_dummies(df, columns=getCategiricals, dummy_na=True)  # one hot encoding of categorical features in dataset
        return df



    def prepareData(self, df):
        """
        prepare data to fit into model
    
        Parameters
        ----------
        df : dataFrame
    
        Returns
        -------
        result : dataFrame
        """
        df = df.replace(r'^\s*$', np.NaN, regex=True)      # conver empty string to Nan
        names = list(filter(lambda x: x not in ['x5', 'x12', 'x31', 'x63', 'x81', 'x82'], df.columns.tolist()))
        for col in names:
            df[col] = df[col].astype("float64")            # convert string to float data type
        result = self.dataPreprocessing(df)
        return result
    
    def convertToDataFrame(self, json):
        """
        convert json to dataFrame
    
        Parameters
        ----------
        json : json
    
        Returns
        -------
        data Frame
    
        """
        if type(json) != list:
            return pd.DataFrame(json, index=[0])
        else:
            return pd.DataFrame(json)      
    

