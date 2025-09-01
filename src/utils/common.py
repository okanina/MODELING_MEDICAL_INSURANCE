import os
import sys
import yaml
import pickle
import json
from datetime import datetime
from ensure import ensure_annotations
from pathlib import Path
from box import ConfigBox
from src.logger import logging
from src.exception import CustomException

TIMESTAMP: datetime = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")

@ensure_annotations
def read_yaml(yaml_path: Path):
    try:
        with open(yaml_path, "r") as yaml_file:
            content = yaml.safe_load(yaml_file)
            return ConfigBox(content)
    except Exception as e:
        raise CustomException(e, sys)

@ensure_annotations
def create_directories(path_to_directories: list):

    try:
        for path in path_to_directories:
            os.makedirs(path, exist_ok=True)
    except Exception as e:
        raise CustomException(e, sys)

def save_obj(file_path, obj):

    """
    This method saves a file to a given path.
    """
    try:
        dir_name=os.path.dirname(file_path)
        os.makedirs(dir_name, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
            
    except Exception as e:
        raise CustomException(e,sys)

@ensure_annotations
def save_json(path:Path, data:dict):
        try:
            with open(path, "w") as f:
                json.dump(data, f)
        except Exception as e:
            raise CustomException(e, sys)

def load_obj(file_path: Path):
    try:
        with open(file_path, "rb") as f:
            return pickle.load(f)

    except Exception as e:
        raise CustomException(e, sys)
    
