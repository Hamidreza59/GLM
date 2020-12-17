#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: hamidrezaouni
"""

import glm_model_26
import pandas as pd
import numpy as np
import unittest
from unittest import TestCase, mock
from helper import ModelHelper
from dataUtil import PrepareData


class TestGLMModel(unittest.TestCase):
    """
    A class for testing glm model 26 application
    """
    def setUp(self):
        # load data
        self.testdata = pd.read_csv("/Users/hamidrezaouni/Downloads/exercise_26_test.csv").head(50)

        
    def test_clean_col(self):
        col_x12 = self.testdata['x12'].head()
        col_x63 = self.testdata['x63'].head()
        new_col_12 = prepareData.clean_x12(self.testdata['x12'].head())
        new_col_63 = prepareData.clean_x63(self.testdata['x63'].head())

        self.assertEqual(new_col_12.dtypes, 'float64')
        self.assertEqual(len(new_col_12), 5)
        self.assertEqual(new_col_63.dtypes, 'float64')
        self.assertEqual(len(new_col_63), 5)
        
    
    def test_normalize(self):
        data = self.testdata['x1'].head(5)
        normal =  (data - data.mean())/data.std()
        normailze = prepareData.normalize(data)
        for ind, val in normal.items():           
            self.assertEqual(normailze[ind], normal[ind])

                         
    def test_feature_selection(self):
        data = self.testdata[['x1', 'x2', 'x3', 'x4', 'x6', 'x7', 'x9', 'x10', 'x11']]
        data = data.fillna(0)                                  #fill 0 for Nan
        data = prepareData.normalize(data)
        y = pd.DataFrame(np.random.randint(2, size=50), columns = ['y'])
        result = helper.feature_selection(data, y, 5)
        self.assertEqual(len(result),5)
        for item in result:
            assert item in data.columns
            
    def test_convertToDataFrame(self):
        data_1 = {"x1":"0.9", "x2": "one"}
        data_2 = [{"x1":"09", "x2": "one"}, {"x1":"0.8", "x2": "two"}]  
        result_1 = prepareData.convertToDataFrame(data_1)
        result_2 = prepareData.convertToDataFrame(data_2)
        self.assertEqual(result_1['x1'][0], "0.9")
        self.assertEqual(result_2['x2'][1], "two")
        
        
                  

if __name__ == '__main__':
    helper = ModelHelper()
    prepareData = PrepareData()
    unittest.main()