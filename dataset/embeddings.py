import numpy as np

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
