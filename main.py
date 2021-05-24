import dataset.load_w2v_model as w2v
import numpy as np
import dataset.data_preprocessor as Bo_Xu_Ly
import dataset.embeddings as embeddings
import dataset.generating_input as generator
import models.sentence_model as embed
import models.layers as layers
import models.rnn as rnn
import models.token_model as token
import models.sentence_model as embed
import models.layers as layers
import models.rnn as rnn
import models.token_model as token
import evaluate.classification_evaluate as test
import visual.visual as visual
import models.save_model.save_and_load as save_load
temp3=None 
model=None
ans=True
url=None
X_Train, X_Test, Y_Train, Y_Test =None,None,None,None
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
    ans=input("What would you like to do?" )
    if ans=="1":
      url=input("Nhap dia chi data: ")        
    elif ans=="2":
      temp=w2v.Load_w2v_model()
      temp.load_w2v()
    elif ans=="3":
      if url is not None:
        A=Bo_Xu_Ly.DuLieu('data/train_test_data/IMDB_Dataset.csv','data/stop_words.txt')
        X_Train, X_Test, Y_Train, Y_Test = A.DATA_PROPROCESSOR_SPLIT()

    elif ans=="4":
      tokenizer = generator.Tokenizer()
      tokenizer.fit_on_texts(X_Train)

      embedding = embeddings.Embeddings("data/stop_words.txt", 300)
      embedding_matrix = embedding.create_embedding_matrix(tokenizer, len(tokenizer.word_counts))

      max_len = np.max([len(text.split()) for text in X_Train])
      TextToTensor_instance = generator.TextToTensor(tokenizer=tokenizer, max_len=max_len)
      X_Train = TextToTensor_instance.string_to_tensor(X_Train) 
    elif ans=="5":
      
      model = layers.Sequential()
      embedding_layer = embed.Embedding(embedding_matrix.shape[0], 300, weights=[embedding_matrix], input_length=max_len , trainable=False)
      model.add(embedding_layer)

      model.add(rnn.LSTM(128))
      model.add(token.Dense(1, activation='sigmoid'))
      model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
    elif ans=="6":
      review_movie_model=model.fit(X_Train, Y_Train, batch_size=256, epochs=10,validation_split=0.2)
    elif ans=="7":
      review_movie_model_pre=test.Test(model,X_Test,Y_Test,tokenizer)
      print(review_movie_model_pre.predict_testdata())
      visualize=visual.Visualize(A.dataset,A.X,review_movie_model)
      visualize.VisualizeData()
      visualize.VisualizePredictModel()
      visualize.VisualizePredictModel2()
    elif ans=="8":
      temp3=save_load.Save_Load(model)
      temp3.save_model("models/save_model")
    elif ans=="9":
      vietem=temp3.load_model("models/save_model")
      vietem.summary()
    else:
      ans=False





