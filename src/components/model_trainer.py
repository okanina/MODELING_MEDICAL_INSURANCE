import sys
import numpy as np
from src.logger import logging
from src.exception import CustomException
from src.entity.artifact_entity import DataTransformationArtifact
from src.entity.config_entity import ModelTrainerConfig
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
from src.entity.artifact_entity import ModelTrainerArtifact
from src.utils.common import *
from xgboost import XGBRegressor

from src.utils.common import save_obj

class ModelTrainer:
    def __init__(self, 
                artifact: DataTransformationArtifact, 
                config: ModelTrainerConfig):

        self.artifact = artifact
        self.config = config

    def initiate_model_trainer(self)->ModelTrainerArtifact:

        report ={}

        try:
            train_arr = np.load(self.artifact.transformed_train_file_path)
           
            X_train, y_train = (train_arr[:,:-1], 
                               train_arr[:, -1]                                           
                                )

            models = self.config.models                       
            params = self.config.param
                                 
            for i in range(len(list(models.values()))):

                model = eval(list(models.values())[i])
                param = params[list(models.keys())[i]]                           
                print(f"Param: {param}")          
                gs = GridSearchCV(estimator = model, param_grid=param, cv=3, return_train_score=True)
                gs.fit(X_train, y_train)

                model.set_params(**gs.best_params_)
                model.fit(X_train, y_train)

                y_train_pred = model.predict(X_train)
                train_score = r2_score(y_train, y_train_pred)

                report[model] = train_score
                
            logging.info("Model Training Completed.")

            best_model_score = max(list(report.values())) 

            if best_model_score < 0.7:
                raise Exception("No model found.")
                                 
            best_model_name = list(report.keys())[list(report.values()).index(best_model_score)]
            
            logging.info(f"The best model found is {best_model_name} has been found.")

            save_obj(self.config.trained_model_file_path, best_model_name)

            model_trainer_artifact = ModelTrainerArtifact(trained_model_file_path =self.config.trained_model_file_path)

            return model_trainer_artifact                               
                
        except Exception as e:
            raise CustomException(e, sys)
