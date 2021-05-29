from keras_preprocessing import sequence
from tensorflow.keras.preprocessing.text import Tokenizer as K
from collections import OrderedDict, defaultdict
maketrans = str.maketrans

def pad_sequences(sequences, maxlen=None, dtype='int32',padding='pre', truncating='pre', value=0.):
    """
        
        Biến đổi danh sách chuỗi thành mãng numpy 2d
        Parameters
        ----------
        sequences : list
            List mãng.
        maxlen : integer
            chiều dài tối đa của các chuỗi.
        dtype:
        The default is 'int32'
        value: integer, float, string
        Giá trị đệm vào khi độ dài chuỗi không đủ maxlen. The default is '0'

        Returns
        -------
        Mãng numpy có độ dài (len(sequences),maxlen)

        """
    return sequence.pad_sequences(
      sequences, maxlen=maxlen, dtype=dtype,
      padding=padding, truncating=truncating, value=value)

def text_to_word_sequence(text,
                          filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                          lower=True, split=" "):
    """
        Parameters
        ----------
        text : string
            List mãng.
        filters : list
            Các ký tự cần loại bỏ.
        lower: boolean
        Chuyển về chữ thường. The default is 'True'
        split: string
        Cắt từ khi gặp split. The default is ' '

        Returns
        -------
        Một danh sách các từ

        """
    if lower:
            text = text.lower()
    
    translate_dict = {c: split for c in filters}
    translate_map = maketrans(translate_dict)
    text = text.translate(translate_map)

    seq = text.split(split)
    return [i for i in seq if i]

class Tokenizer:
    def __init__(self, num_words=None,filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',lower=True,split=' ', char_level=False,oov_token=None,analyzer=None):
        """
        1 nhánh con trong tập xử lý của NLP, là một thuật toán có nhiệm vụ tách từ, cụm từ trong văn bản.

        Parameters
        ----------
        num_words : int , optional
            số lượng từ tối đa cần giữ lại, dựa trên tần suất từ. Chỉ những từ thông dụng nhất sẽ được giữ lại.. The default is None.
        filters : string , optional
            một chuỗi trong đó mỗi phần tử là một ký tự sẽ được lọc khỏi văn bản. Mặc định là tất cả các dấu câu, cộng với tab và ngắt dòng, trừ đi 'ký tự.. The default is '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'.
        lower : boolean, optional
             Có chuyển đổi văn bản thành chữ thường hay không.. The default is True.
        split : string , optional
             Dấu phân tách để tách từ.. The default is ' '.
        char_level : boolean, optional
            nếu True, mọi ký tự sẽ được coi là một mã thông báo.. The default is False.
        oov_token : TYPE, optional
            DESCRIPTION. The default is None.
        analyzer : hàm, optional
            nếu được cung cấp, nó sẽ được thêm vào word_index và được sử dụng để thay thế các từ không có từ vựng trong các cuộc gọi text_to_sequence. The default is None.

        Returns
        -------
        None.

        """

        self.word_counts = OrderedDict()
        self.word_docs = defaultdict(int)
        self.filters = filters
        self.split = split
        self.lower = lower
        self.num_words = num_words
        self.document_count = 0
        self.char_level = char_level
        self.oov_token = oov_token
        self.index_docs = defaultdict(int)
        self.word_index = {}
        self.index_word = {}
        self.analyzer = analyzer
    
    def fit_on_texts(self, texts):
        """
        Cập nhật từ vựng dựa trên danh sách các văn bản.

        Trong trường hợp văn bản chứa danh sách, chúng tôi giả định mỗi mục nhập của danh sách là một mã thông báo.

        Parameters
        ----------
        texts : string
            ó thể là một danh sách các chuỗi, một trình tạo chuỗi (để tiết kiệm bộ nhớ) hoặc một danh sách các chuỗi..

        Returns
        -------
        None.

        """
        for text in texts:
            self.document_count += 1
            if self.char_level or isinstance(text, list):
                if self.lower:
                    if isinstance(text, list):
                        text = [text_elem.lower() for text_elem in text]
                    else:
                        text = text.lower()
                seq = text
            else:
                if self.analyzer is None:
                    seq = text_to_word_sequence(text,
                                                filters=self.filters,
                                                lower=self.lower,
                                                split=self.split)
                else:
                    seq = self.analyzer(text)
            for w in seq:
                if w in self.word_counts:
                    self.word_counts[w] += 1
                else:
                    self.word_counts[w] = 1
            for w in set(seq):
                # In how many documents each word occurs
                self.word_docs[w] += 1

        wcounts = list(self.word_counts.items())
        wcounts.sort(key=lambda x: x[1], reverse=True)
        # forcing the oov_token to index 1 if it exists
        if self.oov_token is None:
            sorted_voc = []
        else:
            sorted_voc = [self.oov_token]
        sorted_voc.extend(wc[0] for wc in wcounts)

        # note that index 0 is reserved, never assigned to an existing word
        self.word_index = dict(
            zip(sorted_voc, list(range(1, len(sorted_voc) + 1))))

        self.index_word = {c: w for w, c in self.word_index.items()}

        for w, c in list(self.word_docs.items()):
            self.index_docs[self.word_index[w]] = c

            
    def texts_to_sequences(self, texts):
        """
        Chuyển từng văn bản trong văn bản thành một chuỗi các số nguyên.

        Chỉ texts - những từ phổ biến nhất mới được tính đến. Chỉ những từ đã biết bởi tokenizer mới được tính đến.

        Parameters
        ----------
        texts : Một danh sách các văn bản
            String .

        Returns
        -------
        TYPE
            Một danh sách các chuỗi .

        """
        return list(self.texts_to_sequences_generator(texts))
    
    def texts_to_sequences_generator(self, texts):
        """
        Chuyển đổi từng văn bản texts thành một chuỗi các số nguyên.

        Mỗi mục trong văn bản cũng có thể là một danh sách,

        Chỉ num_words-1những từ phổ biến nhất mới được tính đến. Chỉ những từ đã biết bởi tokenizer mới được tính đến.

        Parameters
        ----------
        texts : Một danh sách các văn bản 
            list string .

        Yields
        ------
        vect : 
            Mang lại các chuỗi riêng lẻ..

        """
        num_words = self.num_words
        oov_token_index = self.word_index.get(self.oov_token)
        for text in texts:
            if self.char_level or isinstance(text, list):
                if self.lower:
                    if isinstance(text, list):
                        text = [text_elem.lower() for text_elem in text]
                    else:
                        text = text.lower()
                seq = text
            else:
                if self.analyzer is None:
                    seq = text_to_word_sequence(text,
                                                filters=self.filters,
                                                lower=self.lower,
                                                split=self.split)
                else:
                    seq = self.analyzer(text)
            vect = []
            for w in seq:
                i = self.word_index.get(w)
                if i is not None:
                    if num_words and i >= num_words:
                        if oov_token_index is not None:
                            vect.append(oov_token_index)
                    else:
                        vect.append(i)
                elif self.oov_token is not None:
                    vect.append(oov_token_index)
            yield vect

class TextToTensor():
    """ 
    Lớp chuyển đổi từ text sang số và tạo thành ma trận
    """
    def __init__(self, tokenizer, max_len):
        """
        

        Parameters
        ----------
        tokenizer : TYPE
            lớp tokenizer .
        max_len : int 
            đầu ra của lớp tensor .

        Returns Một đối tượng có kiểu có Tensorchức năng chuyển đổi đã đăng ký .
        -------
        None.

        """
        self.tokenizer = tokenizer
        self.max_len = max_len

    def string_to_tensor(self, string_list: list) -> list:
        """
        Hàm chuyển đổi từ text sang vecto
        
        Parameters
        ----------
        string_list : list 
            List chuỗi về  .


        Returns Một đối tượng có kiểu có Tensorchức năng chuyển đổi đã đăng ký .
        -------
        None.
        
        """    
        string_list = self.tokenizer.texts_to_sequences(string_list)
        string_list = pad_sequences(string_list, maxlen=self.max_len)
        
        return string_list
