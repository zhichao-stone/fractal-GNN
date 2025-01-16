import os
import logging

class ModelLogger:
    def __init__(self, name, log_dir:str) -> None:
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)

        file_handler = logging.FileHandler(os.path.join(log_dir, f"{name}.log"))
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

        self.logger.addHandler(file_handler)
        self.logger.addHandler(stream_handler)

    def get_logger(self):
        return self.logger