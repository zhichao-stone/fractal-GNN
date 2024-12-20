import os
import logging
from logging.handlers import TimedRotatingFileHandler

class ModelLogger:
    def __init__(self, name, log_dir:str, backupCount:int=7) -> None:
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)

        file_handler = TimedRotatingFileHandler(os.path.join(log_dir, f"{name}.log"), when="midnight", interval=1, backupCount=backupCount)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

        self.logger.addHandler(file_handler)
        self.logger.addHandler(ch)

    def get_logger(self):
        return self.logger