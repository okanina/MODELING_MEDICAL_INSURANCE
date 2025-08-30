import os
import logging
from datetime import datetime

log_filename=f"{datetime.now().strftime('%d_%m_%Y_%H_%M_%S')}.log"
log_dir_path = os.path.join(os.getcwd(), "logs")
os.makedirs(log_dir_path, exist_ok=True)

file_path = os.path.join(log_dir_path , log_filename)


logging.basicConfig(
    filename=file_path,
    format="[%(asctime)s]- %(name)s -%(levelname)s - %(lineno)d - %(message)s",
    level=logging.INFO
)