# -*- coding: utf-8 -*-
"""
Created on Thu May 27 21:38:39 2021

@author: vieta
"""

from pre_train.pre_input import data_preprocessor as Data, embeddings as emb, generating_input as gen,load_w2v_model as w2v   
import glob
import numpy as np
import pandas as pd
class Pre_train:
    def __init__(self,url=None,stop_word=None):
        self.url=url
        self.stop_word=None
        self.data_after=None
        self.glove=None 
        self.path_glove=None
        self.X_Train, self.X_Test, self.Y_Train, self.Y_Test=None,None,None,None
        self.embedding=None
        self.embedding_matrix=None 
        self.tokenizer=None
        self.max_len=None 
    def Input_url(self,url,path_folder):
        print("Bạn yêu cầu file ở địa  :{}".format(path_folder+url))
        if len(glob.glob(path_folder+url, recursive=True) )>0:
            print("File data  ở :{}".format(path_folder+url))
            self.url=path_folder+url
            return False 
        else:
            return True  
    def Get_Dataset(self):
        return pd.read_csv(self.url)
    def Read_Data(self):
        self.data_after=Data.DuLieu(self.url,self.stop_word)
    def Split_data(self):
        self.X_Train, self.X_Test, self.Y_Train, self.Y_Test=self.data_after.DATA_PROPROCESSOR_SPLIT()
    def Read_glove(self):
        self.glove=w2v.Load_w2v_model()
    def Read_Embdding(self, path_glove=None):
        self.tokenizer = gen.Tokenizer()
        self.tokenizer.fit_on_texts(self.X_Train)
        self.embedding = emb.Embeddings(self.path_glove, 300)
        self.embedding_matrix = self.embedding.create_embedding_matrix(self.tokenizer, len(self.tokenizer.word_counts))
        self.max_len = np.max([len(text.split()) for text in self.X_Train])
        TextToTensor_instance = gen.TextToTensor(tokenizer=self.tokenizer, max_len=self.get_max_len())
        self.X_Train = TextToTensor_instance.string_to_tensor(self.X_Train) 
    def get_max_len(self):
        return self.max_len
    def get_embedding(self):
        return self.embedding_matrix
    def get_X_Train(self):
        return self.X_Train
    def get_Y_Train(self):
        return self.Y_Train
    def get_X_Test(self):
        return self.X_Test
    def get_Y_Test(self):
        return self.Y_Test
    def get_tokenizer(self):
        return self.tokenizer
    def TienXuLy(self,path_folder,path_stop,path_glove):
        self.stop_word=path_stop
        self.path_glove=path_glove
        url =input("Nhập địa chỉ path: ")
        while self.Input_url(url,path_folder):
            print("Bạn đã nhập sai địa chỉ file")
            url =input("Nhập địa chỉ path: ")
        self.Read_Data()
        self.Split_data()
        self.Read_Embdding(self.path_glove)
        

        

 