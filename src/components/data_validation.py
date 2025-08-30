import json
import sys
import pandas as pd
from pandas import DataFrame
from src.exception import CustomException
from src.logger import logging
from src.config.configuration import DataValidationConfig
from src.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from evidently.future.report import Report
from evidently.future.metrics import *
from evidently.future.presets import *
from evidently.future.presets import DataDriftPreset
from evidently.future.datasets import Dataset
from evidently.future.datasets import DataDefinition


class DataValidation:
    def __init__(self, config: DataValidationConfig,
                       artifact:DataIngestionArtifact):
        
        self.config = config
        self.artifact = artifact

        
    def validate_column_schema(self, df:DataFrame)->bool:

        """
        Method Name : validate_column_schema

        Description : This method checks if there is a missing column.
        
        Output      : Returns a boolean value.

        On Failure : It writes an exception log then raise an exception.
        """        
        try:
            
            validate_status=None

            all_columns = df.columns.to_list()
            all_schema = self.config.ALL_SCHEMA
            
            
            for col in all_columns:
                if (col not in all_schema.keys()):
                    validate_status =False                   
                else:
                    validate_status=True

            logging.info("Column validation complete.")

            return validate_status
        
        except Exception as e:
            raise CustomException(e, sys)

    def detect_data_drift(self, reference_df: DataFrame, current_df:DataFrame)->bool:

        """
        Method Name : detect_data_drift

        Description : This method checks if the distribution of incoming data is still the same as the distribution of train dataset..
        
        Output      : Returns a boolean value.

        On Failure : It writes an exception log then raise an exception.
        """

        try:               

            schema= DataDefinition(numerical_columns=self.config.numerical_columns, categorical_columns=self.config.categorical_columns)
            
            eval_data_1 = Dataset.from_pandas(pd.DataFrame(current_df), data_definition=schema)    
            eval_data_2 = Dataset.from_pandas(pd.DataFrame(reference_df), data_definition=schema) 

            logging.info("Detecting data drift.")      

            report = Report(metrics=[DataDriftPreset()],
                                     include_tests=True)
            my_eval = report.run(eval_data_1, eval_data_2)
            report = my_eval.json()
            json_report = json.loads(report)
            
            for key in json_report['tests']:
                if json_report['tests'][key]['status'] == "SUCCESS":                                         
                   validate_status=False          
                else:
                    validate_status=True

            return validate_status

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_validation(self)->DataValidationArtifact:

        """
        Method Name : initiate_data_validation

        Description : This method initiate data validation and .
        
        Output      : Returns an artifact.

        On Failure : It writes an exception log then raise an exception.
        """

        try:  

            train_df = pd.read_csv(self.artifact.train_file_path)
            test_df = pd.read_csv(self.artifact.test_file_path)            

            train_validate_status= self.validate_column_schema(train_df)
            test_validate_status = self.validate_column_schema(test_df)

            drift_status = self.detect_data_drift(train_df, test_df)
            
            if (train_validate_status is True and 
               test_validate_status is True and
               drift_status is False):

               validate_status= True

               with open(self.config.STATUS_FILE, "w") as f:
                f.write(f"Validation Status is {validate_status}")

            else:

                validate_status = False
                with open(self.config.STATUS_FILE, "w") as f:
                    f.write(f"Validation Status is {validate_status}")
                        
            logging.info(f"Validation status is {validate_status}")
            
            data_validation_artifact = DataValidationArtifact(STATUS_FILE= self.config.STATUS_FILE)

            return data_validation_artifact
        
        except Exception as e:
            raise CustomException(e, sys)

