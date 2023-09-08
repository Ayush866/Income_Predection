import logging
import sys,os
import pandas as pd
from income.exception import CustomException
from income.utils import load_object
from income.predictor import ModelResolver
import sklearn

PREDICTION_DIR = "prediction"

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            os.makedirs(PREDICTION_DIR, exist_ok=True)
            logging.info(f"Creating model resolver object")
            model_resolver = ModelResolver(model_registry="saved_models")
            logging.info(f"Reading file :{features}")

            logging.info(f"Loading transformer to transform dataset")
            transformer = load_object(file_path=model_resolver.get_latest_transformer_path())
            model = load_object(file_path=model_resolver.get_latest_model_path())

            input_arr = transformer.transform(features)

            prediction = model.predict(input_arr)

            return prediction


        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    def __init__(self,
                 age: float,
                 workclass: str,
                 education_num:int,
                 marital_status: str,
                 occupation: str,
                 relationship: str,
                 race: str,
                 sex:str,
                 capital_gain:int,
                 capital_loss:int,
                 hours_per_week:int,
                 country:str):


        self.age = age

        self.workclass = workclass

        self.education_num = education_num

        self.marital_status = marital_status

        self.occupation = occupation

        self.relationship = relationship

        self.race = race

        self.sex = sex

        self.capital_gain = capital_gain

        self.capital_loss = capital_loss

        self.hours_per_week = hours_per_week

        self.country = country


    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "age": [self.age],
                "workclass": [self.workclass],
                "education-num": [self.education_num],
                "marital-status": [self.marital_status],
                "occupation": [self.occupation],
                "relationship": [self.relationship],
                "race": [self.race],
                "sex": [self.sex],
                "capital-gain": [self.capital_gain],
                "capital-loss": [self.capital_loss],
                "hours-per-week": [self.hours_per_week],
                "country": [self.country],

            }

            df = pd.DataFrame(custom_data_input_dict)
            # Remove leading and trailing spaces from all columns
            df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
            return df

        except Exception as e:
            raise CustomException(e, sys)