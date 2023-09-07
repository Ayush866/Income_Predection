from income.predictor import ModelResolver
from income.utils import load_object
import pickle
import pandas as pd
from income.pipeline.prediction import CustomData,PredictPipeline
#model_resolver = ModelResolver(model_registry="saved_models")
#transformer = load_object(file_path=model_resolver.get_latest_transformer_path())
#model = load_object(file_path=model_resolver.get_latest_model_path())



list=[49,"Private",5,'Married-spouse-absent',"Other-service","Not-in-family","Black",'Female',0,0,16,'Jamaica']

df= pd.DataFrame([list])
df.columns=["age","workclass","education-num","marital-status","occupation","relationship","race","sex","capital-gain","capital-loss","hours-per-week","country"]
#print(df)
#49, Private,5, Married-spouse-absent, Other-service, Not-in-family, Black, Female,0,0,16, Jamaica
#predict_pipeline = PredictPipeline()
#results = predict_pipeline.predict(df)
#print(results)

#print(model_resolver)
transformer = pickle.load(open("D:/Income_Predection/saved_models/0/transformer/transformer.pkl","rb"))
df = transformer.transform(df)
pickled_model = pickle.load(open('D:/Income_Predection/saved_models/0/model/model.pkl', 'rb'))
if pickled_model.predict(df)==1:
    print("Less")
else:
    print("great")