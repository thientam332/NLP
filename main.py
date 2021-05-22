# -*- coding: utf-8 -*-
"""
Created on Mon May 10 20:21:55 2021

@author: Thiên
"""

import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from keras.preprocessing.text import Tokenizer
from keras.layers import Embedding, LSTM
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers.core import Dropout, Dense
import matplotlib.pyplot as plt

# =============================================================================
# Add Module Data 

# import dataset.data_preprocessor as Bo_Xu_Ly
# A=Bo_Xu_Ly.DuLieu("data//train_test_data//IMDB_Dataset.csv",'data//stop_words.txt')
# X_Train, X_Test, Y_Train, Y_Test = A.DATA_PROPROCESSOR_SPLIT()
# 

# =============================================================================

# =============================================================================
# 
# Read Data
dataset = pd.read_csv("..//data//IMDB Dataset.csv")
dataset.isnull().values.any()
# Trực quan hóa tập dữ liệu theo Label
import seaborn as sns
sns.countplot(x='sentiment', data=dataset)
# Clean text
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

corpus = []
for review in dataset.values[:, 0]:
    review = clean_text(review)
    corpus.append(review)
# One hot Encoder Label
label = LabelEncoder()
Y = label.transform(dataset.iloc[:,1])
# Chia dataset thành train test data
X_Train, X_Test, Y_Train, Y_Test = train_test_split(corpus, Y, test_size=0.20, random_state=40)

# =============================================================================
# Embeddings
# ====================================================================================
# -------------------------------------Đã_add-----------------------------------------
class Embeddings():
    """
    Lớp đọc file word embedding và tạo ma trận dựa trên file đó
    """

    def __init__(self, path, vector_dimension):
        self.path = path 
        self.vector_dimension = vector_dimension
    
    @staticmethod
    # Chuyển đầu vào thành một mảng
    def get_coefs(word, *arr): 
        return word, np.asarray(arr, dtype='float32')
    # Sử dụng dụng file word embedding có sẵn tạo thành từ điển
    def get_embedding_index(self):
        embeddings_index = dict(self.get_coefs(*o.split(" ")) for o in open(self.path, errors='ignore'))
        return embeddings_index
    # Tạo ma trận
    def create_embedding_matrix(self, tokenizer, max_features):
        """
        Hàm tạo ma trận embedding
        """
        model_embed = self.get_embedding_index()

        embedding_matrix = np.zeros((max_features + 1, self.vector_dimension))
        for word, index in tokenizer.word_index.items():
            if index > max_features:
                break
            else:
                try:
                    embedding_matrix[index] = model_embed[word]
                except:
                    continue
        return embedding_matrix
    
class TextToTensor():
    """ 
    Lớp chuyển đổi từ text sang số và tạo thành ma trận
    """
    def __init__(self, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.max_len = max_len

    def string_to_tensor(self, string_list: list) -> list:
        """
        Hàm chuyển đổi từ text sang vecto
        """    
        string_list = self.tokenizer.texts_to_sequences(string_list)
        string_list = pad_sequences(string_list, maxlen=self.max_len)
        
        return string_list

# Mã hóa văn bản
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_Train)

# Tạo Ma Trận Embeddings
embedding = Embeddings("..//data//glove.42B.300d.txt", 300)
embedding_matrix = embedding.create_embedding_matrix(tokenizer, len(tokenizer.word_counts))

# Tạo đầu vào cho mô hình
max_len = np.max([len(text.split()) for text in X_Train])
TextToTensor_instance = TextToTensor(tokenizer=tokenizer, max_len=max_len)
X_Train = TextToTensor_instance.string_to_tensor(X_Train)
#=====================================================================================
# Tạo node Mạng bằng Sequential
model = Sequential()
embedding_layer = Embedding(embedding_matrix.shape[0], 300, weights=[embedding_matrix], input_length=max_len , trainable=False)
model.add(embedding_layer)
model.add(LSTM(128))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
review_movie_model=model.fit(X_Train, Y_Train, batch_size=256, epochs=3,validation_split=0.2)

# Train model
review_movie_model = model.fit(X_Train, Y_Train, batch_size=256, epochs=7)

# Predict test data
TextToTensor_instance = TextToTensor(tokenizer=review_movie_model.tokenizer,max_len=20)
X_Test = [clean_text(text) for text in X_Test]
X_Test = TextToTensor_instance.string_to_tensor(X_Test)
Y_Test = label.fit_transform(Y_Test)
score = review_movie_model.evaluate(X_Test, Y_Test,verbose=0)

# Trực quan hóa kết quả
plt.plot(review_movie_model.history['acc'])
plt.plot(review_movie_model.history['val_acc'])

plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train','test'], loc='upper left')
plt.show()

plt.plot(review_movie_model.history['loss'])
plt.plot(review_movie_model.history['val_loss'])

plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','test'], loc='upper left')
plt.show()