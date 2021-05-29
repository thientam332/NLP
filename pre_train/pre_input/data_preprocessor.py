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
    Làm sạch dữ liệu  thông qua 6 phương pháp:
        -Xóa những định dạng url thông qua thư viện re để xóa những từ https?://\S+|www\.\S+
        -Xóa những định dạng html thông qua thư viện re để xóa những từ <.*?>', '
        -Xóa những định dạng html thông qua thư viện punctuations để xóa những dấu câu 
        -Chuyển về chữ thường 
        -Xóa những định dạng thông qua file stopword yêu cầu để xóa những từ không có ý nghĩa chính cho câu
        -Xóa khoảng trắng thông qua thư viện re
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
    Dùng để lưu trữ dữ liệu và những file yêu cầu trong việc huấn luyện mô hình máy học và học sâu 
    Tiền xử lý dữ liệu , one hot coding label và chia tập train và test 
    """
    def __init__(self,url=None,url_stop=None):
        """
    Khởi tạo class DuLieu 
        Parameters
        ----------
        url : Địa chỉ csv , Kiểu dữ liệu : string 
            url là địa chỉ của bộ dữ liệu  . The default is None 
        stop_stop : File stop word , Kiểu dữ liệu : DataFrame  
            url stop là  file stop word  dùng để chứa những tập dữ liệu từ không có ý nghĩa cho câu
            Dùng thư viện pandas để đọc từ địa chỉ file stop_word được truyền vào 
            . The default is None.
        dataset : Lưu trữ dữ liệu Kiểu dữ liệu : DataFrame  
        Để đọc file csv tạo thành bộ dữ liệu , được đọc thông qua thư viện pandas , gồm nhiều cột và nhiều dòng 
        X : Chứa tập dữ liệu thành phần , Kiểu dữ liệu : list 
        Dùng để chứa tập dữ liệu các thành phần trong dataset 
        Y: Chứa tập dữ liệu nhãn, Kiểu dữ liệu : list 
        Dùng để chứa tập nhãn của dataset 
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
          Method clean_data để khởi tạo bộ dữ liệu dạng thông qua bộ dữ liệu đã cho 
         Tạo ra một list để kiểm soát những phần tử trong dataset
         Sau đó sử dụng function clean_data để xử lý từ trong câu theo bộ dữ liệu
         
          
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
        One-hot encoding là một quá trình mà các biến phân loại (label) 
        được chuyển đổi thành một mẫu có thể cung cấp cho các thuật toán ML để thực hiện công việc tốt hơn khi mà dự đoán.
        Sử dụng LabelEncoder() để mã hóa label 
        Mã hóa label từ text sang number 

        Returns
        -------
        None.

        """
        label = LabelEncoder()
        self.Y = label.fit_transform(self.dataset.iloc[:,1])
        
    
    def Split_Train_Test(self):
        """
        Chia tập train và test theo tỉ lệ train-text: 8:2
        Sử dụng tỉ lệ 8:2 phù hợp cho việc huấn luyện mô hình 
        Và khi chia train-text, được random 40 lần thông qua biến random_state

        Returns
        -------
        TYPE : Tuple bao gồm 4 giá trị, X_train,X_test,Y_train,Y_test 
            Giá trị train và test.

        """
        return train_test_split(self.X, self.Y, test_size=0.20, random_state=40)
        
    def DATA_PROPROCESSOR_SPLIT(self):
        """
        Tiền xử lý tất cả dữ liệu bao gồm các hàm của class Dulieu
        Sử dụng clean_data để xử lý đầu vào của dữ liệu không còn các dữ liệu xấu
        Sử dụng One_Hot_Encoder để mã hóa đầu vào của label 
        Sử dụng Split để chia tập train và test theo tỉ lệ 8:2
        Returns
        -------
        TYPE : Tuple bao gồm 4 giá trị, X_train,X_test,Y_train,Y_test 
                Giá trị train và test.


        """
        self.clean_data()
        self.One_hot_Encoder()
        return self.Split_Train_Test()

    

                   

