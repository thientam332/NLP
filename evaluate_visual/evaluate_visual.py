from visual import visual
from evaluate import evaluate
class Evaluate_visual():
    def __init__(self,model=None,tokenizer=None,dataset=None,X=None,Y=None):
        self.model=model
        self.tokenizer=tokenizer
        self.dataset=dataset
        self.X=X
        self.Y=Y
    def Evaluate(self):
        review_movie_model_pre=evaluate.Test(self.model,self.X,self.Y,self.tokenizer)
        return review_movie_model_pre.predict_testdata()
    def Visuallize(self):
        visualize=visual.Visualize(self.dataset,self.X,self.model)
        visualize.VisualizeData()
        visualize.VisualizePredictModel()
        visualize.VisualizePredictModel2()