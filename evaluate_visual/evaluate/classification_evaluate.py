import dataset.generating_input as generator
class Test():
    """
    Danh gia model 
    """
    def __init__(self,model=None,X=None,Y=None,tokenizer=None):
        """
        

        Parameters
        ----------
        model : class model, optional
            Model cần được đánh giá . The default is None.
        X : Các thành phần của bộ dữ liệu , optional
             The default is None.
        Y : Các label của bộ dữ liệu , optional
            DESCRIPTION. The default is None.
        tokenizer : class , optional
            class chứa các tách từ . The default is None.

        Returns
        -------
        None.

        """
        self.model=model 
        self.X=X
        self.Y=Y
        self.tokenizer=tokenizer
    def predict_testdata(self):
        """
        
        
        Returns
        -------
        score : các giá trị của model được đánh giá 
            độ chính xác của dữ liệu.

        """
        TextToTensor_instance = generator.TextToTensor(tokenizer=self.tokenizer,max_len=20)
        self.X = TextToTensor_instance.string_to_tensor(self.X)
        score = self.model.evaluate(self.X, self.Y,verbose=0)
        return score
    def predict_cau(self):
        TextToTensor_instance = generator.TextToTensor(tokenizer=self.tokenizer,max_len=20)
        self.X = TextToTensor_instance.string_to_tensor(self.X)
        return self.model.predict( self.X)
        