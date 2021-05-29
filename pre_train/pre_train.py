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
    """
    Lớp đầu vào của mô hình
    Dùng để xử lý đầu vào mô hình LSTM thông qua 3 phương thức 
    -Chọn lọc từ và tiền xử lý dữ liệu
    -Tạo các embedding matrix cho bộ dữ liệu
    -Chia tập dữ liệu thành 2 phần train và test theo tỉ lệ 8:2
    """
    def __init__(self,url=None,stop_word=None):
        """
        Khởi tạo lớp đầu vào cho mô hình huấn luyện 

        Parameters
        ----------
        url : TYPE, string 
            Chứa địa chỉ file csv được lưu trữ trong thiết bị. The default is None.
        stop_word : TYPE, DataFrame 
            chứa file txt những từ dư không có ý nghĩa trong câu. The default is None.
        self.data_after : TYPE, pandas
        Chứa dữ liệu sau khi qua phương pháp tiền xử lý trong câu. The default is None.
        self.glove : TYPE, DataFrame  
        Chứa dữ liệu matrix Glove đã được truyền. The default is None.
        self.path_glove: TYPE, string   
        Chứa dữ liệu dữ liệu glove để tạo ma trận các  mối liên hệ, sự tương đồng về mặt ngữ nghĩa. The default is None.
        
        self.X_Train, self.X_Test, self.Y_Train, self.Y_Test TYPE, list   
        Chứa kiểu dữ liệu thành phần và nhãn của tập train và test. The default is None.
        self.embedding TYPE, class ma trận embedding    
        Không gian này bao gồm nhiều chiều và các từ trong không gian đó mà có cùng văn cảnh hoặc ngữ nghĩa sẽ có vị trí gần nhau
        The default is None.
        self.embedding_matrix TYPE, string   
        Chứa dữ liệu dữ liệu glove để tạo ma trận các  mối liên hệ, sự tương đồng về mặt ngữ nghĩa. The default is None.
        self.tokenizer TYPE, class Tokenizer 
        một Class có nhiệm vụ tách từ, cụm từ trong văn bản. The default is None.
        self.max_len TYPE, int 
        Chiều dài tối đa của đầu vào mô hình . The default is None
        Returns
        -------
        None.

        """
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
        """
        Nhập địa chỉ file và kiểm tra trong thư mục có file csv theo đã nhập thông qua thư viện glob
        Nếu có lưu vào địa chỉ file url
        Nếu không trả về false và địa chỉ url = None 

        Parameters
        ----------
        url : string
            Chứa địa chỉ file csv của dataset 
        path_folder : string
            Chứa địa chỉ thư mục của data nơi lưu địa chỉ

        Returns
        -------
        bool
            Giải thích rằng nếu có địa chỉ thì không yêu cầu người dùng nhập lại.

        """
        print("Bạn yêu cầu file ở địa  :{}".format(path_folder+url))
        if len(glob.glob(path_folder+url, recursive=True) )>0:
            print("File data  ở :{}".format(path_folder+url))
            self.url=path_folder+url
            return False 
        else:
            return True  
    def Get_Dataset(self):
        """
        Sau khi có địa chỉ file url thì đọc dữ liệu thông qua pandas 

        Returns
        -------
        DataFrame 
            .Chứa bộ dữ liệu của dataset đã lưu trong project 

        """
        return pd.read_csv(self.url)
    def Read_Data(self):
        """
        Tạo ra dữ liệu sau khi tiền xử lý 

        Returns class Data đã được xử lý 
        -------
        None.

        """
        self.data_after=Data.DuLieu(self.url,self.stop_word)
    def Split_data(self):
        """
        Chia bộ dữ liệu theo 2 tập train và test theo tỉ lệ 8:2 và tiền xử lý theo phương pháp xóa từ trong tập class Data
        Returns 
        -------
        None.

        """
        self.X_Train, self.X_Test, self.Y_Train, self.Y_Test=self.data_after.DATA_PROPROCESSOR_SPLIT()
    def Read_glove(self):
        """
        Đọc kiểu dữ liệu glove theo module load_w2v_model

        Returns
        -------
        None.

        """
        self.glove=w2v.Load_w2v_model()
    def Read_Embdding(self, path_glove=None):
        """
        Tạo ma trận embedding theo các bước:
            Sử dụng class Tokenizer để tách từ và tạo các list phù hợp cho tập train
            Sử dụng method fit lại text trong tập thành phần Train
            Sử dụng class embedding để đọc glove có sẵn trong project 
            Sử dụng ma trận embedding có số chiều max_len để tạo ma trận đầu vào cho mô hình 

        Parameters
        ----------
        path_glove : TYPE, string 
            Địa chỉ file glove . The default is None.

        Returns
        -------
        None.

        """
        self.tokenizer = gen.Tokenizer()
        self.tokenizer.fit_on_texts(self.X_Train)
        self.embedding = emb.Embeddings(self.path_glove, 300)
        self.embedding_matrix = self.embedding.create_embedding_matrix(self.tokenizer, len(self.tokenizer.word_counts))
        self.max_len = np.max([len(text.split()) for text in self.X_Train])
        TextToTensor_instance = gen.TextToTensor(tokenizer=self.tokenizer, max_len=self.get_max_len())
        self.X_Train = TextToTensor_instance.string_to_tensor(self.X_Train) 
    def get_max_len(self):
        """
        Trả về số chiều sử dụng đầu vào trong mô hình

        Returns
        -------
        TYPE : int 
            DESCRIPTION : Sử dụng số chiều cho phép trong glove và giúp việc huấn luyện mô hình tốt hơn.

        """
        return self.max_len
    def get_embedding(self):
        """
        Lấy ma trận embedding cho việc huấn luyện model

        Returns
        -------
        TYPE :  embedding matrix
            Chứa dữ liệu embedding matrix.

        """
        return self.embedding_matrix
    def get_X_Train(self):
        """
        Trả về kiểu dữ liệu tập thành phần dùng để huấn luyện 

        Returns
        -------
        TYPE : list 
            dữ liệu tập thành phần dùng để huấn luyện .

        """
        return self.X_Train
    def get_Y_Train(self):
        """
        
        Trả về kiểu dữ liệu tập nhãn dùng để huấn luyện 
        Returns
        -------
        TYPE : list 
            kiểu dữ liệu tập nhãn dùng để huấn luyện .

        """
        return self.Y_Train
    def get_X_Test(self):
        """
        
        Trả về kiểu dữ liệu tập thành phần dùng để kiểm tra 

        Returns
        -------
        TYPE : list 
            kiểu dữ liệu tập thành phần dùng để kiểm tra  .

        """
        return self.X_Test
    def get_Y_Test(self):
        """
        Trả về kiểu dữ liệu tập nhãn dùng để kiểm tra 
        Returns
        -------
        TYPE : list 
            kiểu dữ liệu tập nhãn dùng để kiểm tra  .

        """
        return self.Y_Test
    def get_tokenizer(self):
        """
        Trả về lớp tokenizer 

        Returns
        -------
        TYPE
            Trả về lớp tách từ của tokenizer .

        """
        return self.tokenizer
    def TienXuLy(self,path_folder,path_stop,path_glove):
        """
        Xử lý đầu vào của mô hình 
            Đọc csv thông qua url
            Đọc stopword cho việc tiền xử lý
            Đọc glove để đầu vào cho ma trận 

        Parameters
        ----------
        path_folder : TYPE : string 
            Địa chỉ thư mục của data.
        path_stop : TYPE : string
            Địa chỉ thư mục của path_stop
        path_glove : TYPE : string 
            Địa chỉ thư mục của path_glove 

        Returns 
        -------
        None.

        """
        self.stop_word=path_stop
        self.path_glove=path_glove
        url =input("Nhập địa chỉ path: ")
        while self.Input_url(url,path_folder):
            print("Bạn đã nhập sai địa chỉ file")
            url =input("Nhập địa chỉ path: ")
        self.Read_Data()
        self.Split_data()
        self.Read_Embdding(self.path_glove)
        

        

 