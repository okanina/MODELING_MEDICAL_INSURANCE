import os
from pathlib import Path

list_of_files =[
    ".github/workflow/.gitkeep",
    "notebooks/exploratory_data_analysis.ipynb",
    "src/__init__.py",
    "src/components/__init__.py",
    "src/components/data_ingestion.py",
    "src/components/data_validaton.py",
    "src/components/data_transformation.py",
    "src/components/model_trainer.py",
    "src/components/model_evaluation.py",
    "src/pipeline/__init__.py",
    "src/pipeline/training_pipeline.py",
    "src/pipeline/prediction_pipeline.py",
    "src/constants/__init__.py",
    "src/entity/__init__.py",
    "src/entity/config_entity.py",
    "src/config/__init__.py",
    "src/config/configuration.py",
    "src/utils/__init__.py",
    "src/utils/common.py",
    "src/config.yaml",
    "src/params.yaml",
    "src/schema.yaml",
    "src/logger.py",
    "src/exception.py",
    "requirements.txt",
    "setup.py"
]

for filepath in list_of_files:
    filepath =Path(filepath)
    filedir, filename = os.path.split(filepath)

    # Creating a directory if it does not exist.
    if filedir !='':
        os.makedirs(filedir, exist_ok=True)
    
    # Creating a file in path if it does not exist.
    if (not os.path.exists(filepath) or (os.path.getsize(filepath)==0)):
        with open(filepath, "w") as f:
            pass