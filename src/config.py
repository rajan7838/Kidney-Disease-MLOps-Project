import os

class Config:
    BASE_DIR = os.getcwd()

    # Paths
    RAW_PATH = os.path.join("artifacts", "raw_data.csv")
    TRAIN_PATH = os.path.join("artifacts", "train.csv")
    TEST_PATH = os.path.join("artifacts", "test.csv")
    MODEL_PATH = os.path.join("models", "best_model.pkl")
    SCALER_PATH = os.path.join("artifacts", "scaler.pkl")

    # Target column
    TARGET_COLUMN = "CKD_Status"

    # MLflow
    MLFLOW_URI = "http://127.0.0.1:5000"
    EXPERIMENT_NAME = "Kidney_Disease_Prediction"