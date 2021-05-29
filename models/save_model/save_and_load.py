# -*- coding: utf-8 -*-
"""
Created on Sat May 22 13:42:51 2021

@author: vieta
"""


import tensorflow.python as tf
class Save_Load():
    def __doc__(self):
        return """
    dùng để lưu model, bao gồm 1 biến : Model
    
    Input:
    Method Init để khởi tạo bộ model 
    
    Output:
    Method để chia tập train và text theo tỉ lệ 8:2
    Method để xử lý toàn bộ quá trình tiền xử lý trả về dạng kiểu dữ liệu tuple bao gồm Train và Text 
    
    Method:
    Method clean_data dùng để chọn lọc từ bao gồm xóa url, thẻ html, chuyển về chữ thường, xóa từ trong stopword,xóa khoảng trắng
    Method One Hot Encode để chuẩn hóa đầu vào Label của thư viện 
    

    """
    def __init__(self,model=None):
        """
        

        Parameters
        ----------
        model : model, optional
            Kiểu model đã có khi chạy. The default is None.

        Returns trả về kiểu 
        -------
        None.

        """
        self.model=model
    def save_model(self,url):
        """
        Dùng để save model 
                Parameters
        ----------
        url : string 
            địa chỉ lưu .

        Returns
        -------
        TYPE
            dữ liệu model đã lưu.

        """
        print(self.model.summary())
        self.model.save(url)
    def load_model(self,url):
        """
        

        Parameters
        ----------
        url : string 
            địa chỉ lưu .

        Returns
        -------
        TYPE
            dữ liệu model đã lưu.

        """
        self.model=  tf.keras.models.load_model(url)
        return  self.model


# a=Save_Load()

# a.load_model("C://Users//vieta/OneDrive - Trường ĐH CNTT - University of Information Technology//Github//NLP//data//my_model")
