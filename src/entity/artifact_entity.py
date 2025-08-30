from dataclasses import dataclass
from pathlib import Path

@dataclass
class DataIngestionArtifact:
    train_file_path: str
    test_file_path: str

@dataclass
class DataValidationArtifact:
    STATUS_FILE : Path    

@dataclass
class DataTransformationArtifact:
    transformed_train_file_path: str
    transformed_test_file_path: str
    transformed_object_file_path: str

@dataclass
class ModelTrainerArtifact:
    trained_model_file_path: str

@dataclass
class ModelEvaluationArtifact:
    metric_file_path: Path