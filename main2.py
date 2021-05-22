# -*- coding: utf-8 -*-


import dataset.data_preprocessor as Bo_Xu_Ly
A=Bo_Xu_Ly.DuLieu('C:/Users/Admin/Documents/GitHub/NLP/data/train_test_data/IMDB_Dataset.csv','C:/Users/Admin/Documents/GitHub/NLP/data/stop_words.txt')
X_Train, X_Test, Y_Train, Y_Test = A.DATA_PROPROCESSOR_SPLIT()

