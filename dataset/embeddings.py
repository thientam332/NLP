import numpy as np

class Embeddings():
    """
    Lớp đọc file word embedding và tạo ma trận dựa trên file đó
    """

    def __init__(self, path, vector_dimension):
        """
        Chuyển đầu vào thành một mảng, và tạo ma trận embedding
        
        Parameters
        ----------
        path : string 
            địa chỉ của lớp embedding .
        vector_dimension : int 
            số chiều của lớp embedding .

        Returns
        -------
        None.

        """
        self.path = path 
        self.vector_dimension = vector_dimension
    
    @staticmethod
    def get_coefs(word, *arr): 
        """
        Chuyển đầu vào thành một mảng

        Parameters
        ----------
        word : strings 
            kiểu dữ liệu của word .
        *arr : list 
         nơi lưu trữ array .

        Returns 
        -------
        TYPE string và kiểu dữ liệu np.array 
            Chuyển đầu vào thành một mảng.

        """
        return word, np.asarray(arr, dtype='float32')
    # Sử dụng dụng file word embedding có sẵn tạo thành từ điển
    def get_embedding_index(self):
        """
        Sử dụng dụng file word embedding có sẵn tạo thành từ điển

        Returns
        -------
        embeddings_index : dict ( ) 
            Từ  của embedding .

        """
        embeddings_index = dict(self.get_coefs(*o.split(" ")) for o in open(self.path, errors='ignore'))
        return embeddings_index
    # Tạo ma trận
    def create_embedding_matrix(self, tokenizer, max_features):
        """
        Tạo ma trận

        Parameters
        ----------
        tokenizer : từ điển đã lưu 
            DESCRIPTION.
        max_features : int số chiều cần lưu 
            DESCRIPTION.

        Returns
        -------
        embedding_matrix : matrix numpy 
            Ma trận biểu hiện mối liên quan .

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
