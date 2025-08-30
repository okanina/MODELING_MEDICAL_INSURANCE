import sys
from pathlib import Path
from src.config.configuration import configurationManager
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.data_validation import DataValidation
from src.components.model_trainer import ModelTrainer
from src.components.model_evaluation import ModelEvaluation
from src.logger import logging
from src.exception import CustomException


class TrainPipeline:
    def __init__(self):

        self.config = configurationManager()        


    def start_data_ingestion(self):

        logging.info(">>>>>>>>>>>Data Ingestion Initiated.<<<<<<<<<<")

        try:
            data_ingestion_config = self.config.get_data_ingestion_config()
            data_ingestion = DataIngestion(config = data_ingestion_config)
            data_ingestion_artifact=data_ingestion.initiate_data_ingestion()

            logging.info(">>>>>>>>>>>Data Ingestion Completed.<<<<<<<<<<")

            return data_ingestion_artifact

        except Exception as e:
            raise CustomException(e, sys)

    def start_data_validation(self, data_ingestion_artifact):

        logging.info(">>>>>>>>>>>Data Validation Initiated.<<<<<<<<<<")

        try:
            data_validation_config = self.config.get_data_validation_config()
            data_validation =DataValidation(config = data_validation_config, artifact = data_ingestion_artifact)
            data_validation_artifact = data_validation.initiate_data_validation()

            logging.info(">>>>>>>>>>>Data Validation Completed.<<<<<<<<<<")
            
            return data_validation_artifact
            
        except Exception as e:
            raise CustomException(e, sys)
        

    def start_data_transformation(self, data_validation_artifact, data_ingestion_artifact):

        logging.info(">>>>>>>>>>>Data Transformation Initiated.<<<<<<<<<<")

        try:
            
            with open(data_validation_artifact.STATUS_FILE, "r") as f:
                status=f.read().split(" ")[-1]

            if status=="True":            
                data_transformation_config = self.config.get_data_transformation_config()
                data_transformation = DataTransformation(config=data_transformation_config, artifact=data_ingestion_artifact)
                data_tranformation_artifact= data_transformation.initiate_data_transformaion()
                
                logging.info(">>>>>>>>>>>Data Transformation ompleted.<<<<<<<<<<")

                return data_tranformation_artifact
                   
            else:
                raise Exception("Your Column data schema is not valid.")
        except Exception as e:
            raise CustomException(e, sys)
 

    def start_model_trainer(self, data_transformation_artifact):

        logging.info(">>>>>>>>>>>Model Trainer Initiated.<<<<<<<<<<")

        try:
            model_trainer_config =self.config.get_model_trainer_config()
            model_trainer =ModelTrainer(config = model_trainer_config, artifact=data_transformation_artifact)
            model_trainer_artifact = model_trainer.initiate_model_trainer()
            logging.info(">>>>>>>>>>>Model Training Completed.<<<<<<<<<<")

            return model_trainer_artifact
        
        except Exception as e:
            raise CustomException(e, sys)
   

    def start_model_evaluation(self, model_trainer_artifact, data_transformation_artifact):
        logging.info(">>>>>>>>>>>Model Evaluation Initiated.<<<<<<<<<<")

        try:
            model_evaluation_config =self.config.get_model_evaluation_config()
            model_evaluation = ModelEvaluation(config =model_evaluation_config,
                                               model_trainer_artifact=model_trainer_artifact,
                                               transformation_artifact=data_transformation_artifact)
            model_evaluation.initiate_model_evaluation()
            
            logging.info(">>>>>>>>>>>Model Evaluation Completed.<<<<<<<<<<")

        except Exception as e:
            raise CustomException(e, sys)
    
    
    def run_pipeline(self)->None:
        logging.info("********************Initiating Train pipeline*****************")

        try:
            data_ingestion_artifact = self.start_data_ingestion()
            data_validation_artifact = self.start_data_validation(data_ingestion_artifact=data_ingestion_artifact)
            data_transformation_artifact = self.start_data_transformation(data_validation_artifact=data_validation_artifact, data_ingestion_artifact=data_ingestion_artifact)
            model_trainer_artifact =self.start_model_trainer(data_transformation_artifact=data_transformation_artifact)
            self.start_model_evaluation(model_trainer_artifact=model_trainer_artifact,
                                        data_transformation_artifact=data_transformation_artifact)
            logging.info("********************Train pipeline Complete.*****************")
        except Exception as e:
            raise CustomException(e, sys)
        
if __name__=="__main__":
    pipeline = TrainPipeline()
    pipeline.run_pipeline()