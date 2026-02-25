# Kidney Disease MLOps Project

End-to-end MLOps pipeline for predicting Chronic Kidney Disease (CKD) using patient data.

## Features
- Modular code with logging and exception handling
- Data ingestion, preprocessing, model training with GridSearchCV
- MLflow experiment tracking
- Streamlit web app for real-time predictions
- Dockerized and CI/CD ready

## Setup
1. Clone repository
2. Install dependencies: `pip install -r requirements.txt`
3. Place dataset in `data/kidney_disease.csv`
4. Run training: `python train.py`
5. Launch app: `streamlit run app.py`

## Deployment
- Build Docker image: `docker build -t kidney-app .`
- Run container: `docker run -p 8501:8501 kidney-app`