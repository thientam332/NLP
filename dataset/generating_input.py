from keras_preprocessing import sequence
from tensorflow.keras.preprocessing.text import Tokenizer as K
from collections import OrderedDict, defaultdict
maketrans = str.maketrans

def pad_sequences(sequences, maxlen=None, dtype='int32',padding='pre', truncating='pre', value=0.):
    return sequence.pad_sequences(
      sequences, maxlen=maxlen, dtype=dtype,
      padding=padding, truncating=truncating, value=value)

def text_to_word_sequence(text,
                          filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                          lower=True, split=" "):
    if lower:
            text = text.lower()
    
    translate_dict = {c: split for c in filters}
    translate_map = maketrans(translate_dict)
    text = text.translate(translate_map)

    seq = text.split(split)
    return [i for i in seq if i]

class Tokenizer:
    def __init__(self, num_words=None,filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',lower=True,split=' ', char_level=False,oov_token=None,analyzer=None):
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
        return list(self.texts_to_sequences_generator(texts))
    
    def texts_to_sequences_generator(self, texts):
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
        self.tokenizer = tokenizer
        self.max_len = max_len

    def string_to_tensor(self, string_list: list) -> list:
        """
        Hàm chuyển đổi từ text sang vecto
        """    
        string_list = self.tokenizer.texts_to_sequences(string_list)
        string_list = pad_sequences(string_list, maxlen=self.max_len)
        
        return string_list
