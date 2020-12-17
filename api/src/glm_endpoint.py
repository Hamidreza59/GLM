#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: hamidrezaouni
"""

import traceback
import pickle
from flask import Blueprint, request, jsonify
from api.src.model.dataUtil import PrepareData


# API definition
glm_api = Blueprint('glm_api', __name__) 
 
 
with open("api/src/model/pickle_model.pkl", "rb") as file:
    model= pickle.load(file)               
    print ('Model loaded')

with open("api/src/model/pickle_variable.pkl", "rb") as file:
    model_columns = pickle.load(file)
    print ('Model columns loaded')

prepare = PrepareData()

@glm_api.route('/predict', methods=['POST'])
def predict():
    """
    return predict API to predict new data set
    Parameters
    ----------
    none
   
    """ 
    
    if model:
        try:
            json_ = request.json            
            data = prepare.convertToDataFrame(json_)
            query = prepare.prepareData(data)
            query = query.reindex(columns=model_columns, fill_value=0)      # just use feature from model for predict
            prediction = list(model.predict(query))
            classes = list(map(lambda x: 1 if x >= 0.5 else 0, prediction))

            return jsonify({'prediction': str(prediction), 'varaibles': model_columns, "classes": classes})

        except:
            
            return jsonify({'trace': traceback.format_exc()})
    else:
        print ('Train the model first')
        return ('There is no model here to use')