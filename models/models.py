# -*- coding: utf-8 -*-
"""
Created on Thu May 27 22:00:14 2021

@author: Admin
"""


from models.lstm import layers, rnn, sentence_model, token_model
class ModelLSTM():
    """
    Lớp xây dựng mô hình LSTM và train mô hình
    """
    def __init__(self,Pre_train=None):
        """
        Nhận dữ liệu train mô hình

        Parameters
        ----------
        Pre_train : object
            Class chứa các tham số đầu vào đã được xử lý. The default is None.
        self.model: object 
            Khởi tạo đối tượng rỗng làm nơi lưu trữ cho quá trình train mô hình. The default is None.
        Returns
        -------
        None.

        """
        self.Pre_train=Pre_train
        self.model=None
    def CreateNode(self):
        """
        Xây dựng mô hình LSTM và train mô hình với các tham số từ Pre_train và các layer Keras

        Returns
        -------
        Object
            Lớp lịch sử của train model.

        """
        self.model = layers.Sequential()
        embedding_layer = sentence_model.Embedding(self.Pre_train.get_embedding().shape[0], 300, weights=[self.Pre_train.get_embedding()], input_length=self.Pre_train.get_max_len() , trainable=False)
        self.model.add(embedding_layer)
        self.model.add(rnn.LSTM(128))
        self.model.add(token_model.Dense(1, activation='sigmoid'))
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
        return self.model.fit(self.Pre_train.get_X_Train(), self.Pre_train.get_Y_Train(), batch_size=256, epochs=10,validation_split=0.2)
    def get_model(self):
        """
        Hàm trả về model đang được thực thi.

        Returns
        -------
        Object
            Model.

        """
        return self.model