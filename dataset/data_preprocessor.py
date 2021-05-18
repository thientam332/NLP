# -*- coding: utf-8 -*-
"""
Created on Mon May 10 20:21:55 2021

@author: Tran Viet Anh

"""
import pandas as pd
import re


from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


stop_words = []
stop_words = pd.read_csv('..//data//stop_words.txt', sep='\n', header=None)[0].tolist()

def clean_text(string: str, punctuations=r'''!()-[]{};:'"\,<>./?@#$%^&*_~''',stop_words=stop_words) -> str:
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
def Preprocessor(url ="..//data//train_test_data//IMDB_Dataset.csv"):
        
    # Read Data
    dataset = pd.read_csv(url)
    dataset.isnull().values.any()
    # Trực quan hóa tập dữ liệu theo Label
    import seaborn as sns
    sns.countplot(x='sentiment', data=dataset)

    corpus = []
    for review in dataset.values[:, 0]:
        review = clean_text(review)
        corpus.append(review)
    # One hot Encoder Label
    # One hot Encoder Label
    label = LabelEncoder()
    Y = label.fit_transform(dataset.iloc[:,1])
    X_Train, X_Test, Y_Train, Y_Test = train_test_split(corpus, Y, test_size=0.20, random_state=40)
    return  X_Train, X_Test, Y_Train, Y_Test 

A=Preprocessor()