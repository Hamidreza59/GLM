#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: hamidrezaouni
"""

from .app import app
 
if __name__ == "__main__":
    app.run(use_reloader=True)