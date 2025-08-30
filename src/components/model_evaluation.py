import sys
import joblib
import pickle
from pathlib import Path
from src.utils.common import save_json
import numpy as np
from src.logger import logging
from src.exception import CustomException
from src.entity.config_entity import ModelEvaluationConfig
from src.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact
from sklearn.metrics import mean_absolute_error, mean_squared_error,r2_score

class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig,
                       model_trainer_artifact : ModelTrainerArtifact,
                       transformation_artifact: DataTransformationArtifact):

        self.config = config
        self.model_trainer_artifact = model_trainer_artifact
        self.transformation_artifact = transformation_artifact

    def evaluation_metric(self, actual, pred):
        mae = mean_absolute_error(actual, pred)
        mse = mean_squared_error(actual, pred)
        r2 = r2_score(actual, pred)

        return mae, mse, r2

    def initiate_model_evaluation(self):

        try:
            test_arr = np.load(self.transformation_artifact.transformed_test_file_path)
            with open(self.model_trainer_artifact.trained_model_file_path, "rb") as f:
                model=pickle.load(f)

            X_test, y_test = (
                              test_arr[:,:-1],
                              test_arr[:, -1]
                             )
            pred =model.predict(X_test)

            (mae, mse, r2) = self.evaluation_metric(y_test, pred)

            scores = {"mae":mae, "mse":mse, "r2":r2}

            logging.info("Model evaluation completed.")

            save_json(Path(self.config.metric_file_path), scores)            
        except Exception as e:
            raise CustomException(e, sys)
