"""This module saves the trained model as a file so it can be used later for prediction or deployment"""



import joblib
import os
from src.config import Config
from src.logging.logger import logging

class ModelPusher:
    def save(self, model):
        os.makedirs("models", exist_ok=True)
        joblib.dump(model, Config.MODEL_PATH)
        logging.info(f"Model saved to {Config.MODEL_PATH}")