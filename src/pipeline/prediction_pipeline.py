import sys
from src.exception import CustomException
import pandas as pd
from src.utils.common import load_obj


class predictPipeline:
    def __iit__(self):
        pass

    def get_predictions(self, features):

        try:
            preprocessor =load_obj("artifact\data_transformation\preprocessor.pkl")
            model = load_obj("artifact\model_trainer\model.pkl")

            scaled_data = preprocessor.transform(features)

            preds = model.predict(scaled_data)

            return preds

        except Exception as e:
            raise CustomException(e, sys)



class CustomData:
    def __init__(self, age, sex, weight, bmi, smoker, diabetes, hereditary_diseases, city, bloodpressure, regular_ex, no_of_dependents, job_title):
        self.age = age
        self.sex = sex
        self.weight = weight
        self.bmi = bmi
        self.diabetes = diabetes
        self.smoker = smoker
        self.hereditary_diseases = hereditary_diseases
        self.city = city
        self.bloodpressure = bloodpressure
        self.regular_ex = regular_ex
        self.no_of_dependents = no_of_dependents
        self.job_title = job_title

    def get_dataframe(self):
        try:
            input_data = {
                "age":[self.age],
                "sex":[self.sex],
                "weight": [self.weight],
                "bmi": [self.bmi],
                "diabetes": [self.diabetes],
                "smoker": [self.smoker],
                "hereditary_diseases": [self.hereditary_diseases],
                "city": [self.city],
                "bloodpressure": [self.bloodpressure],
                "regular_ex": [self.regular_ex],
                "no_of_dependents" : [self.no_of_dependents],
                "job_title" : [self.job_title]
            }            

            return pd.DataFrame(input_data).reset_index()
        except Exception as e:
            raise CustomException(e, sys)
