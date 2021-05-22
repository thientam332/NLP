import dataset.generating_input as generator
class Test():
    def __init__(self,model=None,X=None,Y=None,tokenizer=None):
        self.model=model 
        self.X=X
        self.Y=Y
        self.tokenizer=tokenizer
    def predict_testdata(self):
        TextToTensor_instance = generator.TextToTensor(tokenizer=self.tokenizer,max_len=20)
        self.X = TextToTensor_instance.string_to_tensor(self.X)
        score = self.model.evaluate(self.X, self.Y,verbose=0)
        return score