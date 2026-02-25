"""This file orchestrates the complete end-to-end MLOps pipeline from data ingestion to model saving"""




from src.components.data_ingestion import DataIngestion
from src.components.data_preprocessing import DataPreprocessing
from src.components.model_trainer import ModelTrainer
from src.components.model_evaluation import ModelEvaluation
from src.components.model_pusher import ModelPusher
from src.logging.logger import logging

if __name__ == "__main__":
    logging.info("=== Starting Kidney Disease MLOps Pipeline ===")

    ingestion = DataIngestion()
    train_path, test_path = ingestion.initiate("data/kidney_disease.csv")

    preprocessing = DataPreprocessing()
    X_train, X_test, y_train, y_test = preprocessing.transform(train_path, test_path)

    trainer = ModelTrainer()
    model = trainer.train(X_train, y_train, X_test, y_test)

    evaluator = ModelEvaluation()
    evaluator.evaluate(model, X_test, y_test)

    pusher = ModelPusher()
    pusher.save(model)

    print("Kidney Disease MLOps Pipeline Completed")
    logging.info("=== Pipeline Completed Successfully ===")
