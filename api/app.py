#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: hamidrezaouni

"""

# Dependencies
from flask import Flask
from .src.glm_endpoint import glm_api

# API definition
app = Flask(__name__)
app.register_blueprint(glm_api)
        
    

if __name__ == '__main__':
   app.run(host="0.0.0.0")