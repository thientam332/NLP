# -*- coding: utf-8 -*-
"""
Created on Sat May 22 13:42:51 2021

@author: vieta
"""


import tensorflow.python as tf
class Save_Load():
    def __init__(self,model=None):
      self.model=model
    def save_model(self,url):
      self.model.save(url)
    def load_model(self,url):
      self.model=  tf.keras.models.load_model(url)
      return  self.model


a=Save_Load()

a.load_model("C://Users//vieta/OneDrive - Trường ĐH CNTT - University of Information Technology//Github//NLP//data//my_model")
