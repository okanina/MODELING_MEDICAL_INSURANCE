from src.logger import logging
from src.exception import CustomException
from src.utils.common import read_yaml, create_directories
from src.entity.config_entity import (DataIngestionConfig, 
                                     DataValidationConfig,
                                     DataTransformationConfig,
                                     ModelTrainerConfig,
                                     ModelEvaluationConfig)
from src.constants import CONFIG_FILE_PATH, PARAMS_FILE_PATH, SCHEMA_FILE_PATH

class configurationManager:
    def __init__(self, 
                config_filepath = CONFIG_FILE_PATH,
                params_filepath= PARAMS_FILE_PATH,
                schema_filepath=SCHEMA_FILE_PATH
                ):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        self.schema =read_yaml(schema_filepath)

        create_directories([self.config.artifact_root])

    def get_data_ingestion_config(self):

        config = self.config.data_ingestion
        params = self.params

        create_directories([config.data_ingestion_dir])

        data_ingestion_config = DataIngestionConfig(
            data_ingestion_dir = config.data_ingestion_dir,
            source_url = config.source_url,
            local_data_file = config.local_data_file,
            train_file_path = config.train_file_path,
            test_file_path =config.test_file_path,
            test_size =params.test_size,
            features = params.features
            )

        return data_ingestion_config
    
    def get_data_validation_config(self):
        config =self.config.data_validation
        schema = self.schema

        create_directories([config.data_validation_dir])

        data_validation_config = DataValidationConfig(data_validation_dir = config.data_validation_dir,
                                                       STATUS_FILE =config.STATUS_FILE,
                                                      ALL_SCHEMA=schema.COLUMNS,
                                                      categorical_columns = schema.categorical_columns,
                                                      numerical_columns=schema.numerical_columns
                                                      )
        
        return data_validation_config

    def get_data_transformation_config(self):

        config = self.config.data_transformation
        schema = self.schema.TARGET_COLUMN

        create_directories([config.transformed_data_dir])

        data_transformation_config = DataTransformationConfig(
                transformed_data_dir = config.transformed_data_dir,
                transformed_train_file_path = config.transformed_train_file_path,
                transformed_test_file_path = config.transformed_test_file_path,
                transformed_object_file_path = config.transformed_object_file_path,
                test_size = self.params.test_size,
                target_column = schema.target_column
                )

        return data_transformation_config
    
    def get_model_trainer_config(self):

        config = self.config.model_trainer
        params = self.params

        create_directories([config.model_trainer_dir])

        model_trainer_config = ModelTrainerConfig(model_trainer_dir = config.model_trainer_dir,
                                                  trained_model_file_path = config.trained_model_file_path,
                                                  models = params.models,
                                                  param =  params.param                                              
                                                 )
        
        return model_trainer_config
    
    def get_model_evaluation_config(self):
        config = self.config.model_evaluation

        create_directories([config.model_evaluation_dir])

        model_evaluation_config = ModelEvaluationConfig(model_evaluation_dir = config.model_evaluation_dir,
                                                       metric_file_path = config.metric_file_path
                                                        )
        
        return model_evaluation_config
