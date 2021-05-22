from urllib.request import urlopen
from zipfile import ZipFile

class Load_w2v_model():
    def __init__(self,url='http://nlp.stanford.edu/data/glove.42B.300d.zip'):
        self.url = url
    
    def load_w2v(self):
        zipresp = urlopen(self.url)
        if (zipresp is not None):
            tempzip = open("C:/Users/Admin/Documents/GitHub/NLP/data/glove.zip", "wb")
            tempzip.write(zipresp.read())
            tempzip.close()
            zf = ZipFile("C:/Users/Admin/Documents/GitHub/NLP/data/glove.zip")
            zf.extractall(path = 'C:/Users/Admin/Documents/GitHub/NLP/data')
            zf.close()
        else:
            print("Url không tồn tại")