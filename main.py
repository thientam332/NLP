#Package
from config import *
from dataset import data_preprocessor as Data,embeddings as embed, generating_input as gene ,load_w2v_model as w2v 

from models import layers,rnn,sentence_model as sen,token_model as token 
from visual import visual as vs
from models.save_model import save_and_load  as sv 
from evaluate import classification_evaluate as  ev
#Module
import pandas as pd
import numpy as np

url=None
while ans:
    print (""""
    1. Input path_Data
    2. Download File Glove 
    3. Tiền xử lý dữ liệu, Chia Data
    4. Tạo lớp Embedding 
    5. Thiết lập Model
    6. Thực hiện quá trình huấn luyện
    7. Trực quan hóa dữ liệu
    8. Lưu model 
    9. Load model 
    """)
    print("""
          1. Tiền xử lý dữ liệu và chuẩn bị đầu vào cho model
          2. Xây dựng lớp mạng LSTM và train mô hình
          3. Trực quan hóa
          4. Save Model
          5. Load Model
          """)
    ans=input("What would you like to do?" )
    if ans=="1":
        while url!=Path_data:
            url=input("Nhap dia chi data: ")
            if url==Path_data:
                print("Bạn đã nhập địa chỉ file csv")
            else:
                print("Bạn đã nhập sai địa chỉ")
    elif ans=="2":
        temp=w2v.Load_w2v_model()
    elif ans=="3":
        if url!=None:
            A=Data.DuLieu(url,Path_stop_words)
            X_Train, X_Test, Y_Train, Y_Test = A.DATA_PROPROCESSOR_SPLIT()
        else:
            print("Data chạy theo mặc định")
            A=Data.DuLieu(Path_data,Path_stop_words)
            X_Train, X_Test, Y_Train, Y_Test = A.DATA_PROPROCESSOR_SPLIT()
    elif ans=="4":        
      tokenizer = gene.Tokenizer()
      tokenizer.fit_on_texts(X_Train)

      embedding = embed.Embeddings(Path_glove, 300)
      embedding_matrix = embedding.create_embedding_matrix(tokenizer, len(tokenizer.word_counts))

      max_len = np.max([len(text.split()) for text in X_Train])
      TextToTensor_instance = gene.TextToTensor(tokenizer=tokenizer, max_len=max_len)
      X_Train = TextToTensor_instance.string_to_tensor(X_Train) 
      
    elif ans=="5":
      from models import models
      model=models.ModelLSTM(data)
      review_movie_model=model.CreateNode()
    elif ans=="6":
      from evaluate_visual import evaluate_visual
      val_visual=evaluate_visual.Evaluate_visual(model.get_model,data.get_tokenizer(),data.get_X_Test(),data.get_Y_Test())
      val_visual.Evaluate()
      val_visual.Visuallize()
      
    elif ans=="7":
      review_movie_model_pre=ev.Test(model,X_Test,Y_Test,tokenizer)
      print(review_movie_model_pre.predict_testdata())
      visualize=vs.Visualize(A.dataset,A.X,review_movie_model)
      visualize.VisualizeData()
      visualize.VisualizePredictModel()
      visualize.VisualizePredictModel2()
    elif ans=="8":
      temp3=sv.Save_Load(model)
      temp3.save_model(Path_model)
    elif ans=="9":
      vietem=temp3.load_model(Path_model)
      vietem.summary()



