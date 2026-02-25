"""This module evaluates the trained model and saves performance metrics in a JSON file for tracking and monitoring"""



from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from src.logging.logger import logging
import json
import os

class ModelEvaluation:
    def evaluate(self, model, X_test, y_test):
        preds = model.predict(X_test)
        accuracy = accuracy_score(y_test, preds)
        report = classification_report(y_test, preds, output_dict=True)
        cm = confusion_matrix(y_test, preds).tolist()

        metrics = {
            "accuracy": accuracy,
            "classification_report": report,
            "confusion_matrix": cm
        }

        os.makedirs("artifacts", exist_ok=True)
        with open("artifacts/final_metrics.json", "w") as f:
            json.dump(metrics, f, indent=4)

        logging.info(f"Evaluation completed. Accuracy: {accuracy}")
        return metrics
        