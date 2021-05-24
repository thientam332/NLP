from wordcloud import WordCloud
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from IPython import get_ipython
class Visualize():
    def __init__(self,dataset,X=None,model=None):
        
        self.dataset=dataset
        self.X=X
        self.model=model
        """
        Dùng để trực quan hóa dữ liệu
        Input: method init để khởi tạo bộ dữ liệu dạng csv
        Output: Đồ thị trực quan hóa dataset, Accuracy và loss
        """
    def VisualizeData(self):
        """
        Trực quan hóa dataset
        Kiểm tra tần suất xuất hiện của 1 từ, từ đó xuất hiện càng nhiều thì kích cỡ chữ càng lớn
        """
        sns.countplot(x='sentiment', data=self.dataset)
        plt.show()
        get_ipython().run_line_magic('matplotlib', 'inline')
        cloud = np.array(self.X).flatten()
        plt.figure(figsize=(20,10))
        word_cloud = WordCloud(max_words=100,background_color ="black",
                                       width=2000,height=1000,mode="RGB").generate(str(cloud))
        plt.axis("off")
        plt.imshow(word_cloud)
    def VisualizePredictModel(self):
        """
        Trực quan hóa Accuracy sau khi train model
        Kiểm tra độ chính xác của model
        """
        plt.plot(self.model.history['acc'])
        plt.plot(self.model.history['val_acc'])
        
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train','test'], loc='upper left')
        plt.show()
        
    def VisualizePredictModel2(self):
        """
        Trực quan hóa Loss sau khi train model
        Kiểm tra độ mất mát của model
        """
        plt.plot(self.model.history['loss'])
        plt.plot(self.model.history['val_loss'])
        
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train','test'], loc='upper left')
        plt.show()