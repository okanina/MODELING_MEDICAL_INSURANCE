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
            
            # models = {
            #           'LinearRegression': LinearRegression(),
            #           'Ridge': Ridge(),
            #           'Lasso': Lasso(),
            #           'DecisionTreeRegressor': DecisionTreeRegressor(),
            #           'RandomForestRegressor': RandomForestRegressor(),
            #           'AdaBoostRegressor': AdaBoostRegressor(),
            #         #   'KNeighborsRegressor': KNeighborsRegressor(),
            #         # 'XGBRegressor': XGBRegressor()
            #          }
            # params = {'LinearRegression': {},
            #           'Ridge': {'alpha': [1.0, 2.0, 3.0]},
            #           'Lasso': {},
            #           'DecisionTreeRegressor': {'criterion': ['squared_error','friedman_mse', 'absolute_error', 'poisson'],
            #                                   'splitter': ["best", "random"],
            #                                   'max_depth': range(100,300,100) },
            #           'RandomForestRegressor':{'max_features': [0.25,0.5,0.75,1.0],
            #                                  'max_depth': range(50,200,50),
            #                                  'n_estimators': range(50,200,50)
            #                                  },
            #           'AdaBoostRegressor': {'n_estimators': range(50,200,50),
            #                               'learning_rate': [.001, .01, .05, .1],
            #                               'random_state': [1]},
            #         #   'KNeighborsRegressor':{'n_neighbors': range(5,15,5),
            #         #                       'weights': ['distance', 'uniform']
            #         #                       },
                    #   'XGBRegressor': {'learning_rate': [0.1, 0.01, 0.05, 0.001],
                    #                  'n_estimators': range(100, 300, 100),
                    #                  'max_depth' : range(3,9,3)
                    #                 } 
            # }

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

                report[list(models.keys())[i]] = train_score
                logging.info(f"Done training {list(models.values())[i]}")

            logging.info("Model Training Completed.")

            best_model_score = max(list(report.values())) 

            if best_model_score < 0.7:
                raise Exception("No model found.")
                                 
            best_model_name = list(report.keys())[list(report.values()).index(best_model_score)]
            best_model= self.config.models[best_model_name]

            logging.info(f"The best model found is {best_model} has been found.")

            save_obj(self.config.trained_model_file_path, best_model)

            model_trainer_artifact = ModelTrainerArtifact(trained_model_file_path =self.config.trained_model_file_path)

            return model_trainer_artifact                               
                
        except Exception as e:
            raise CustomException(e, sys)
