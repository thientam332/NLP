from urllib.request import urlopen
from zipfile import ZipFile

class Load_w2v_model():
    """
    Dùng chứa kiểu dữ liệu matrix glove có sẵn
    Sử dụng thư viện tại địa chỉ 'http://nlp.stanford.edu/data/glove.42B.300d.zip'
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
        Truy cập 'http://nlp.stanford.edu/data/glove.42B.300d.zip' thông qua urlopen của thư viện urlib.request
        Sử dụng method write với file được tạo trong thư mục data 
        Input: self.url

        Returns
        -------
        file word embedding đã được giải nén.

        """
        zipresp = urlopen(self.url)
        if (zipresp is not None):
            tempzip = open("data/glove.zip", "wb")
            tempzip.write(zipresp.read())
            tempzip.close()
            zf = ZipFile("data/glove.zip")
            zf.extractall(path = 'data')
            zf.close()
        else:
            print("Url không tồn tại")