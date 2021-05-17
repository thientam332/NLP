from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

def pad_sequences():
    pass

class Tokenizer():
    def __init__(self):
        pass
    def fit_on_texts(data):
        pass
    def texts_to_sequences():
        pass

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
