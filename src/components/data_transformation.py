import os
import sys
import pandas as pd
import numpy as np
from src.logger import logging
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.exception import CustomException
from sklearn.model_selection import train_test_split
from src.entity.config_entity import DataTransformationConfig
from src.entity.artifact_entity import DataIngestionArtifact, DataTransformationArtifact
from src.components.data_ingestion import DataIngestion
from src.utils.common import save_obj

class DataTransformation:
    def __init__(self, config:DataTransformationConfig,
                       artifact: DataIngestionArtifact):

            self.config = config
            self.artifact = artifact
            
            
    def initiate_data_transformaion(self)->DataTransformationArtifact:
        """
        Method Name : split_data_to_train_test

        Description : This method split the dataset into train and test set.

        Output : Creates a folder in 

        On Failure : It writes an exception log then raise an exception.

        """

        try:
            train_df = pd.read_csv(self.artifact.train_file_path)
            test_df = pd.read_csv(self.artifact.test_file_path)
            
            train_input = train_df.drop(columns=[self.config.target_column])
            train_target = train_df[self.config.target_column]
            
            test_input = test_df.drop(columns=[self.config.target_column])
            test_target = test_df[self.config.target_column]

            logging.info("Initialized standardscaler and simpleimputer")

            num_pipeline = Pipeline(
                steps =[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )  

            cat_pipeline = Pipeline(
                steps =[
                    ("ohe", OneHotEncoder(sparse_output=False, handle_unknown="ignore"))
                ]

            )   

            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, train_input.select_dtypes("number").columns.to_list()),
                    ("cat_pipeline",cat_pipeline, train_input.select_dtypes("object").columns.to_list())
                ]
            )  

            transformed_train_input = preprocessor.fit_transform(train_input)
            transformed_test_input = preprocessor.transform(test_input)

            train_arr = np.c_[transformed_train_input, np.array(train_target)]
            test_arr = np.c_[transformed_test_input, np.array(test_target)]

            logging.info("Column transformation complete.")

            np.save(self.config.transformed_train_file_path.replace("csv", "npy"), train_arr)
            np.save(self.config.transformed_test_file_path.replace("csv", "npy"), test_arr)
            save_obj(self.config.transformed_object_file_path, preprocessor)
            
            logging.info("train_arr, test_arr and preprocessor successfully saved.")

            data_tranformation_artifact = DataTransformationArtifact(transformed_train_file_path =self.config.transformed_train_file_path,
                                                                     transformed_test_file_path = self.config.transformed_test_file_path,
                                                                     transformed_object_file_path = self.config.transformed_object_file_path
                                                                    )
            
            return data_tranformation_artifact
        except Exception as e:
            raise CustomException(e, sys)
