"""This module reads raw data, saves it, splits into train-test, and 
stores them in artifacts with proper logging and exception handling"""


import pandas as pd
import os
from sklearn.model_selection import train_test_split
from src.logging.logger import logging
from src.exception.custom_exception import CustomException
from src.config import Config
import sys

class DataIngestion:
    def initiate(self, data_path):
        try:
            logging.info("Data Ingestion Started")
            df = pd.read_csv(data_path)

            os.makedirs("artifacts", exist_ok=True)
            df.to_csv(Config.RAW_PATH, index=False)

            train, test = train_test_split(df, test_size=0.2, random_state=42)
            train.to_csv(Config.TRAIN_PATH, index=False)
            test.to_csv(Config.TEST_PATH, index=False)

            logging.info("Data Ingestion Completed")
            return Config.TRAIN_PATH, Config.TEST_PATH

        except Exception as e:
            raise CustomException(e, sys)







