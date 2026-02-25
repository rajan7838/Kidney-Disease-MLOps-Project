"""This ModelTrainer class trains multiple machine learning models using GridSearchCV for hyperparameter tuning. 
It logs parameters, metrics, and models into MLflow for experiment tracking. Finally, 
it selects and returns the best performing model based 
on test accuracy."""

import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from src.config import Config
from src.logging.logger import logging

class ModelTrainer:
    def train(self, X_train, y_train, X_test, y_test):
        mlflow.set_tracking_uri(Config.MLFLOW_URI)
        mlflow.set_experiment(Config.EXPERIMENT_NAME)

        models = {
            "RandomForest": (
                RandomForestClassifier(random_state=42),
                {"n_estimators": [100, 200], "max_depth": [None, 10]}
            ),
            "DecisionTree": (
                DecisionTreeClassifier(random_state=42),
                {"max_depth": [None, 10, 20], "min_samples_split": [2, 5]}
            ),
            "GradientBoosting": (
                GradientBoostingClassifier(random_state=42),
                {"n_estimators": [100, 200], "learning_rate": [0.01, 0.1]}
            )
        }

        best_model = None
        best_accuracy = 0

        for name, (model, params) in models.items():
            with mlflow.start_run(run_name=name):
                logging.info(f"Training {name} with GridSearchCV")
                grid = GridSearchCV(model, params, cv=5, scoring='accuracy')
                grid.fit(X_train, y_train)

                best_estimator = grid.best_estimator_
                accuracy = best_estimator.score(X_test, y_test)

                mlflow.log_params(grid.best_params_)
                mlflow.log_metric("test_accuracy", accuracy)
                mlflow.sklearn.log_model(best_estimator, name)

                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_model = best_estimator

        logging.info(f"Best model accuracy: {best_accuracy}")
        return best_model