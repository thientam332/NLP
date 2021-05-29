import numpy as np

class Embeddings():
    """
    Lớp đọc file word embedding và tạo ma trận dựa trên file đó
    """

    def __init__(self, path, vector_dimension):
        """
        
        Khởi tạo class Embeddings một không gian vector 
        Dùng để biểu diễn dữ liệu có khả năng miêu tả được mối liên hệ, sự tương đồng về mặt ngữ nghĩa, văn cảnh(context) của dữ liệu. 
        Không gian này bao gồm nhiều chiều và các từ trong không gian đó mà có cùng văn cảnh hoặc ngữ nghĩa sẽ có vị trí gần nhau.
        Parameters
        ----------
        path : kiểu dữ liệu của file embedding  TYPE: string 
            Đường dẫn tới nơi lưu trữ file word embedding trên local.
            
        vector_dimension : integer
            Số chiều của vector cần thiết khi khởi tạo ma trận embedding .

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
        Hàm tĩnh Chuyển đầu vào thành một mảng
        Xử lý trên mảng list không bị thiếu dữ liệu và thuận tiện cho việc sử dụng matrix embedding
        
        
        Parameters
        ----------
        word : string
        Tập dữ liệu các thành phần để được chuẩn hóa thông qua embedding 
        *arr : string
            Tham số thể hiện độ ánh xạ ngữ nghĩa.

        Returns
        -------
        word :từ
        và biến có kiểu numpy có các thành phần float 

        """
        return word, np.asarray(arr, dtype='float32')
    # Sử dụng dụng file word embedding có sẵn tạo thành từ điển
    def get_embedding_index(self):
        """
        Sử dụng dụng file word embedding có sẵn tạo thành từ điển
        Mở File glove (ma trận có sẵn thông qua địa chỉ file )
        tách từ trong file bằng split xóa khoảng trắng , và dùng get_coefs để đưa về kiểu dữ liệu list
        Sử dụng dict để tạo từ điển trong class

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
        Tạo ra ma trận đầu vào đồng thời sử dụng tokenizer và giá trị max_features để tạo ra ma trận
        Nếu giá trị trong tokenizer nhỏ hơn max_feature sẽ được chuẩn hóa về embedding_matrix 

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
