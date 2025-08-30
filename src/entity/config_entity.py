from dataclasses import dataclass
from pathlib import Path

@dataclass
class DataIngestionConfig:
    data_ingestion_dir: Path
    source_url: str
    local_data_file: Path
    train_file_path: Path
    test_file_path: Path
    test_size: float
    features: list

@dataclass
class DataValidationConfig:
    data_validation_dir: Path
    STATUS_FILE : str
    ALL_SCHEMA:dict
    categorical_columns: list
    numerical_columns:list      

@dataclass
class DataTransformationConfig:
    transformed_data_dir: str
    transformed_train_file_path: str
    transformed_test_file_path: str
    transformed_object_file_path: str    
    target_column: str 
    test_size: float

@dataclass
class ModelTrainerConfig:
    model_trainer_dir: str
    trained_model_file_path: str
    models: dict
    param : dict    

@dataclass
class ModelEvaluationConfig:
    model_evaluation_dir: str
    metric_file_path: Path
