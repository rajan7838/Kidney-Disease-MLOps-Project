"""This preprocessing step separates features and target, removes unnecessary columns, 
applies standard scaling to normalize feature values, saves the scaler for deployment, and returns 
transformed training and testing datasets"""

import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from src.config import Config

class DataPreprocessing:
    def transform(self, train_path, test_path):
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)

        target = Config.TARGET_COLUMN
        # Drop target and dialysis_needed from features
        X_train = train_df.drop([target, 'Dialysis_Needed'], axis=1)
        y_train = train_df[target]

        X_test = test_df.drop([target, 'Dialysis_Needed'], axis=1)
        y_test = test_df[target]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        joblib.dump(scaler, Config.SCALER_PATH)
        return X_train_scaled, X_test_scaled, y_train, y_test


