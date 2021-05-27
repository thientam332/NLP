# -*- coding: utf-8 -*-
"""
Created on Thu May 27 22:00:14 2021

@author: Admin
"""


from lstm import layers, rnn, sentence_model, token_model
class ModelLSTM():
    def __init__(self,Pre_train=None):
        self.Pre_train=Pre_train
        self.model=None
    def CreateNode(self):
        self.model = layers.Sequential()
        embedding_layer = sentence_model.Embedding(self.Pre_train.get_embedding().shape[0], 300, weights=[self.Pre_train.get_embedding()], input_length=self.Pre_train.get_max_len() , trainable=False)
        self.model.add(embedding_layer)
        self.model.add(rnn.LSTM(128))
        self.model.add(token_model.Dense(1, activation='sigmoid'))
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
        return self.model.fit(self.Pre_train.get_X_Train(), self.Pre_train.get_Y_Train(), batch_size=256, epochs=10,validation_split=0.2)
    def get_model(self):
        return self.model