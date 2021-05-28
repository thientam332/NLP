# #Package
# from config import *
# from dataset import data_preprocessor as Data,embeddings as embed, generating_input as gene ,load_w2v_model as w2v 

# from models import layers,rnn,sentence_model as sen,token_model as token 
# from visual import visual as vs
# from models.save_model import save_and_load  as sv 
# from evaluate import classification_evaluate as  ev
# #Module
# import pandas as pd
# import numpy as np
from config import *
from pre_train import pre_train as pt
from models import models
from evaluate_visual import evaluate_visual as ev
from models.save_model import save_and_load as sl
url=None

while ans:
    print("""
          1. Tiền xử lý dữ liệu và chuẩn bị đầu vào cho model
          2. Xây dựng lớp mạng LSTM và train mô hình
          3. Trực quan hóa
          4. Save Model
          5. Load Model
          """)
    ans=input("What would you like to do?" )
    if ans=="1":
        dulieu=pt.Pre_train()

        dulieu.TienXuLy(Path_data,Path_stop_words,Path_glove)
    elif ans=="2":
        MODEL=models.ModelLSTM(dulieu)
        MODEL.CreateNode()
    elif ans=="3":
        VISUAL=ev.Evaluate_visual(MODEL.get_model(),dulieu.get_tokenizer(),dulieu.Get_Dataset(),dulieu.get_X_Test(),dulieu.get_Y_Test())
        print("Evaluate: ")
        print(VISUAL.Evaluate())
        VISUAL.Visuallize()
    elif ans=="4":
        SAVE=sl.Save_Load(MODEL.get_model())
        path=input("Bấm 0 để chọn địa chỉ mặc định hoặc Nhap dia chi save: ")
        if path=="0":
            path=Path_model
        SAVE.save_model(path)
    elif ans=="5":
        LOAD=sl.Save_Load()
        path=input("Bấm 0 để chọn địa chỉ mặc định hoặc Nhap dia chi save: ")
        if path=="0":
            path=Path_model
        LOAD.load_model(path)

    else:
        ans=False
        
        
        

       
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
    # elif ans=="4":        
    #   tokenizer = gene.Tokenizer()
    #   tokenizer.fit_on_texts(X_Train)

    #   embedding = embed.Embeddings(Path_glove, 300)
    #   embedding_matrix = embedding.create_embedding_matrix(tokenizer, len(tokenizer.word_counts))

    #   max_len = np.max([len(text.split()) for text in X_Train])
    #   TextToTensor_instance = gene.TextToTensor(tokenizer=tokenizer, max_len=max_len)
    #   X_Train = TextToTensor_instance.string_to_tensor(X_Train) 
      
    # elif ans=="5":
    #   from models import models
    #   model=models.ModelLSTM(data)
    #   review_movie_model=model.CreateNode()
    # elif ans=="6":
    #   from evaluate_visual import evaluate_visual
    #   val_visual=evaluate_visual.Evaluate_visual(model.get_model,data.get_tokenizer(),data.get_X_Test(),data.get_Y_Test())
    #   val_visual.Evaluate()
    #   val_visual.Visuallize()
      
    # elif ans=="7":
    #   review_movie_model_pre=ev.Test(model,X_Test,Y_Test,tokenizer)
    #   print(review_movie_model_pre.predict_testdata())
    #   visualize=vs.Visualize(A.dataset,A.X,review_movie_model)
    #   visualize.VisualizeData()
    #   visualize.VisualizePredictModel()
    #   visualize.VisualizePredictModel2()
    # elif ans=="8":
    #   temp3=sv.Save_Load(model)
    #   temp3.save_model(Path_model)
    # elif ans=="9":
    #   vietem=temp3.load_model(Path_model)
    #   vietem.summary()



