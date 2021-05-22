# -*- coding: utf-8 -*-
import numpy as np

import dataset.data_preprocessor as Bo_Xu_Ly
A=Bo_Xu_Ly.DuLieu('C:/Users/Admin/Documents/GitHub/NLP/data/train_test_data/IMDB_Dataset.csv','C:/Users/Admin/Documents/GitHub/NLP/data/stop_words.txt')
X_Train, X_Test, Y_Train, Y_Test = A.DATA_PROPROCESSOR_SPLIT()

# =============================================================================
# import dataset.load_w2v_model as w2v
# temp=w2v.Load_w2v_model()
# temp.load_w2v()
# =============================================================================

import dataset.embeddings as embeddings

import dataset.generating_input as generator
tokenizer = generator.Tokenizer()
tokenizer.fit_on_texts(X_Train)

embedding = embeddings.Embeddings("C:/Users/Admin/Documents/GitHub/NLP/data/glove.42B.300d.txt", 300)
embedding_matrix = embedding.create_embedding_matrix(tokenizer, len(tokenizer.word_counts))

max_len = np.max([len(text.split()) for text in X_Train])
TextToTensor_instance = generator.TextToTensor(tokenizer=tokenizer, max_len=max_len)
X_Train = TextToTensor_instance.string_to_tensor(X_Train)



import models.sentence_model as embed
import models.layers as layers
import models.rnn as rnn
import models.token_model as token

model = layers.Sequential()
embedding_layer = embed.Embedding(embedding_matrix.shape[0], 300, weights=[embedding_matrix], input_length=max_len , trainable=False)
model.add(embedding_layer)
model.add(rnn.LSTM(128))
model.add(token.Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
review_movie_model=model.fit(X_Train, Y_Train, batch_size=256, epochs=10,validation_split=0.2)

import models.save_model.save_and_load as save_load
temp3=save_load.Save_Load(model)
temp3.save_model("C:/Users/Admin/Documents/GitHub/NLP/models/save_model")

vietem=temp3.load_model("C:/Users/Admin/Documents/GitHub/NLP/models/save_model")

import evaluate.classification_evaluate as test
review_movie_model_pre=test.Test(model,X_Test,Y_Test,tokenizer)
print(review_movie_model_pre.predict_testdata())

import visual.visual as visual
visualize=visual.Visualize(A.dataset,A.X,review_movie_model)
visualize.VisualizeData()
visualize.VisualizePredictModel()
visualize.VisualizePredictModel2()

