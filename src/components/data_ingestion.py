import os
import sys
import kaggle
import pandas as pd
from src.logger import logging
from src.exception import CustomException
from src.entity.config_entity import DataIngestionConfig
from src.entity.artifact_entity import DataIngestionArtifact
from sklearn.model_selection import train_test_split


class DataIngestion:
    def __init__(self, config:DataIngestionConfig):

        self.config = config

    def initiate_data_ingestion(self)->DataIngestionArtifact:

        """
        Method Name : initiate_data_ingestion

        Description : This method downloads data from the kaggle.

        On Failure : It writes an exception log then raise an exception.
        """
        
        try:
            kaggle.api.authenticate()

            if not os.path.exists(self.config.local_data_file):
                kaggle.api.dataset_download_files(self.config.source_url, 
                                                path=self.config.data_ingestion_dir, unzip=True)

                logging.info("Dataset downloaded.")
            else:
                logging.info("Data file path already exist.")
            
            df = pd.read_csv(self.config.local_data_file)

            for feature in self.config.features:
                df[feature]=df[feature].replace({0:'no',1:'yes'})
                df[feature].astype("object")               

            logging.info(f"The shape of the dataset: {df.shape}")

            train_df, test_df = train_test_split(df, 
                                                   test_size = self.config.test_size, 
                                                   random_state = 42)
            
            logging.info(f"Train Test Split complete.Train shape: {train_df.shape}. Test shape: {test_df.shape}")

            train_df.to_csv(self.config.train_file_path, index=False)
            test_df.to_csv(self.config.test_file_path, index=False)

            logging.info("train_df and test_df have been successfully saved.")
            
            data_ingestion_artifact = DataIngestionArtifact(train_file_path = self.config.train_file_path,
                                                            test_file_path =self.config.test_file_path)

            return data_ingestion_artifact
                      
        except Exception as e:
            raise CustomException(e, sys)

