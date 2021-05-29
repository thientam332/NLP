from evaluate_visual.visual import visual
from evaluate_visual.evaluate import classification_evaluate as evaluate
class Evaluate_visual():
    """
    Lớp trực quan hóa và đánh giá kết quả
    """
    def __init__(self,model=None,tokenizer=None,dataset=None,X=None,Y=None,History=None ):
        """
        Nhận và khởi tạo các dữ liệu để trực quan hóa và đánh giá mô hình.

        Parameters
        ----------
        model : Object
            model đã train. The default is None.
        tokenizer : Object
            Class được khởi tạo từ thư viện Tokenizer của keras. Dùng để xử lý Testdata trước khi dùng để đánh giá. The default is None.
        dataset : DataFrame 
            Bộ dữ liệu của dataset đã lưu trong project . The default is None.
        X : list
            Danh sách các bình luận phim. The default is None.
        Y : list
            Danh sách các nhãn tương ứng của bình luận. The default is None.
        History : Object
            Class lưu trữ lịch sử của model đã train. The default is None.

        Returns
        -------
        None.

        """
        self.model=model
        self.tokenizer=tokenizer
        self.dataset=dataset
        self.X=X
        self.Y=Y
        self.History=History
    def Evaluate(self):
        """
        Hàm đánh giá độ chính xác trên datatest

        Returns
        -------
        dict
             Gồm 2 giá trị loss và acc.
        """
        review_movie_model_pre=evaluate.Test(self.model,self.X,self.Y,self.tokenizer)
        return review_movie_model_pre.predict_testdata()
    def Visuallize(self):
        """
        Hàm trực quan hóa dữ liệu ban đầu và kết quả dự đoán trên tập test với acc và val_acc

        Returns
        -------
        None.

        """
        visualize=visual.Visualize(self.dataset,self.X,self.History)
        visualize.VisualizeData()
        visualize.VisualizePredictModel2()
    def Visuallize1(self):
        """
        Hàm trực quan hóa kết quả dự đoán trên tập test với loss và val_loss

        Returns
        -------
        None.

        """
        visualize1=visual.Visualize(self.dataset,self.X,self.History)
        visualize1.VisualizePredictModel()  
    