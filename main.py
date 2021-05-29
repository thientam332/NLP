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
        HISTORY= MODEL.CreateNode()
    elif ans=="3":
        VISUAL=ev.Evaluate_visual(MODEL.get_model(),dulieu.get_tokenizer(),dulieu.Get_Dataset(),dulieu.get_X_Test(),dulieu.get_Y_Test(),HISTORY)
        print("Evaluate: ")
        print(VISUAL.Evaluate())
        VISUAL.Visuallize1()
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
        temp=LOAD.load_model(path) 
    else:
        ans=False
