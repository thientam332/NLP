from urllib.request import urlopen
from zipfile import ZipFile

class Load_w2v_model():
    def __doc__(self):
        return """
    Input: Url word embedding model
    Output: file word embedding có sẵn
    Tính năng: download file word embedding ở định dạng zip và giải nén.
    """
    def __init__(self,url='http://nlp.stanford.edu/data/glove.42B.300d.zip'):
        """
        Parameters
        ----------
        url : link http
            Đ/c file word embedding online. The default is 'http://nlp.stanford.edu/data/glove.42B.300d.zip'.

        Returns
        -------
        None.

        """
        self.url = url
    
    def load_w2v(self):
        """
        Input: self.url

        Returns
        -------
        file word embedding đã được giải nén.

        """
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