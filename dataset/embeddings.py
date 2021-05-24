import numpy as np

class Embeddings():
    """
    Lớp đọc file word embedding và tạo ma trận dựa trên file đó
    """

    def __init__(self, path, vector_dimension):
        """
        

        Parameters
        ----------
        path : link
            Đường dẫn tới nơi lưu trữ file word embedding trên local.
        vector_dimension : integer
            Số chiều của vecto.

        Returns
        -------
        None.

        """
        self.path = path 
        self.vector_dimension = vector_dimension
    
    @staticmethod
    # Chuyển đầu vào thành một mảng
    def get_coefs(word, *arr): 
        """

        Parameters
        ----------
        word : string
        *arr : string
            Tham số thể hiện độ ánh xạ ngữ nghĩa.

        Returns
        -------
        word :từ
        numpy

        """
        return word, np.asarray(arr, dtype='float32')
    # Sử dụng dụng file word embedding có sẵn tạo thành từ điển
    def get_embedding_index(self):
        """
        

        Returns
        -------
        embeddings_index : dict
            Tạo thành từ điển word - tham số ánh xạ.

        """
        embeddings_index = dict(self.get_coefs(*o.split(" ")) for o in open(self.path, errors='ignore'))
        return embeddings_index
    # Tạo ma trận
    def create_embedding_matrix(self, tokenizer, max_features):
        """
        

        Parameters
        ----------
        tokenizer : class
            class tách từ.
        max_features : integer
            số lượng word đầu vào.

        Returns
        -------
        embedding_matrix : numpy
            Ma trận embedding.
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
