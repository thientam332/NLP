# -*- coding: utf-8 -*-
"""
Created on Wed May 26 10:15:42 2021

@author: vieta
"""

Path_data='data/train_test_data/IMDB_Dataset.csv'
Path_glove='data/glove.42B.300d.txt'
Path_stop_words='data/stop_words.txt'

train_size=0.2

model=None

ans=True
url=None
X_Train, X_Test, Y_Train, Y_Test =None,None,None,None


Path_model = 'models/save_model'
