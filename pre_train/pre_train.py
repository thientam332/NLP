# -*- coding: utf-8 -*-
"""
Created on Thu May 27 21:38:39 2021

@author: vieta
"""

    # if ans=="1":
    #     while url!=Path_data:
    #         url=input("Nhap dia chi data: ")
    #         if url==Path_data:
    #             print("Bạn đã nhập địa chỉ file csv")
    #         else:
    #             print("Bạn đã nhập sai địa chỉ")
    # elif ans=="2":
    #     temp=w2v.Load_w2v_model()
    # elif ans=="3":
    #     if url!=None:
    #         A=Data.DuLieu(url,Path_stop_words)
    #         X_Train, X_Test, Y_Train, Y_Test = A.DATA_PROPROCESSOR_SPLIT()
    #     else:
    #         print("Data chạy theo mặc định")
    #         A=Data.DuLieu(Path_data,Path_stop_words)
    #         X_Train, X_Test, Y_Train, Y_Test = A.DATA_PROPROCESSOR_SPLIT()
      #   elif ans=="4":        
      # tokenizer = gene.Tokenizer()
      # tokenizer.fit_on_texts(X_Train)

      # embedding = embed.Embeddings(Path_glove, 300)
      # embedding_matrix = embedding.create_embedding_matrix(tokenizer, len(tokenizer.word_counts))

      # max_len = np.max([len(text.split()) for text in X_Train])
      # TextToTensor_instance = gene.TextToTensor(tokenizer=tokenizer, max_len=max_len)
      # X_Train = TextToTensor_instance.string_to_tensor(X_Train) 
from pre_input import data_preprocessor as Data, embeddings as emb, generating_input as gen,load_w2v_model as w2v   

import os
import numpy as np

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
    def Input_url(self,url=None):
        while self.url!=None:
            path_folder="../data"
            files = os.listdir(path_folder)    
            files = list(filter(lambda f: f.endswith('.csv'), files))
            if url in files:
                self.url=url
            else:
                print("Bạn đã nhập sai địa chỉ")
    def Read_Data(self):
        self.data_after=Data.DuLieu(self.url,self.stop_word)
    def Split_data(self):
        self.X_Train, self.X_Test, self.Y_Train, self.Y_Test=self.data_after.DATA_PROPROCESSOR_SPLIT()
    def Read_glove(self):
        self.glove=w2v.Load_w2v_model()
    def Read_Embdding(self):
        self.tokenizer = gen.Tokenizer()
        self.tokenizer.fit_on_texts(self.X_Train)
        self.embedding = emb.Embeddings(self.path_glove, 300)
        self.embedding_matrix = self.embedding.create_embedding_matrix(self.tokenizer, len(self.tokenizer.word_counts))
        self.max_len = np.max([len(text.split()) for text in self.X_Train])
        TextToTensor_instance = gen.TextToTensor(tokenizer=self.tokenizer, max_len=self.get_max_len())
        self.X_Train = TextToTensor_instance.string_to_tensor(self.X_Train) 
    def get_max_len(self):
        return np.max([len(text.split()) for text in self.X_Train])
    def get_embedding(self):
        return self.embedding_matrix
    def get_X_Train(self):
        return self.X_Train
    def get_Y_Train(self):
        return self.Y_Train
    def get_tokenizer(self):
        return self.tokenizer
 