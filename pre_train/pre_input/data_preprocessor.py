# -*- coding: utf-8 -*-
"""
Created on Mon May 10 20:21:55 2021

@author: Tran Viet Anh

"""
import pandas as pd
import re


from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

def clean_text(string: str, punctuations=r'''!()-[]{};:'"\,<>./?@#$%^&*_~''',stop_words=None) -> str:
    """
    Làm sạch dữ liệu  
    """
    # Xóa urls
    string = re.sub(r'https?://\S+|www\.\S+', '', str(string))

    # Xóa thẻ html
    string = re.sub(r'<.*?>', '', str(string))

    # Xóa dấu câu
    for x in string.lower(): 
        if x in punctuations: 
            string = string.replace(x, "") 
    # Chuyển về kiểu chữ thường
    string = string.lower()

    # Xóa từ trong stop word
    string = ' '.join([word for word in string.split() if word not in stop_words])

    # Xóa khoảng trắng
    string = re.sub(r'\s+', ' ', str(string)).strip()

    return string

class DuLieu():
    """
    Tiền xử lý dữ liệu , chia file train và test 
    """
    def __init__(self,url=None,url_stop=None):
        """
    
        Parameters
        ----------
        url : Địa chỉ csv
            url là địa chỉ của bộ dữ liệu  . The default is "..//data//train_test_data//IMDB_Dataset.csv".
        url_stop : Địa chỉ stop word
            url stop là địa chỉ của file stop word . The default is '..//data//train_test_data//stop_words.txt'.

        Returns 
        -------
        None.

        """
        self.url=url
        self.dataset=pd.read_csv(url)
        self.stop_words = pd.read_csv(url_stop, sep='\n', header=None)[0].tolist()
        self.X=[]
        self.Y=None 
        
        
    def clean_data(self):
        """
          Method Init để khởi tạo bộ dữ liệu dạng thông qua địa chỉ 
          
        Returns
        -------
        None.

        """
        corpus = []
        for review in self.dataset.values[:, 0]:
            review = clean_text(review,r'''!()-[]{};:'"\,<>./?@#$%^&*_~''',self.stop_words)
            corpus.append(review)
            self.X=corpus

    def One_hot_Encoder(self):
        """
        Mã hóa label từ text sang number 

        Returns
        -------
        None.

        """
        label = LabelEncoder()
        self.Y = label.fit_transform(self.dataset.iloc[:,1])
        
    
    def Split_Train_Test(self):
        """
        Chia tập train và test lại 

        Returns
        -------
        TYPE
            Giá trị train và test.

        """
        return train_test_split(self.X, self.Y, test_size=0.20, random_state=40)
        
    def DATA_PROPROCESSOR_SPLIT(self):
        """
        Tiền xử lý tất cả dữ liệu 

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        self.clean_data()
        self.One_hot_Encoder()
        return self.Split_Train_Test()

    

                   
