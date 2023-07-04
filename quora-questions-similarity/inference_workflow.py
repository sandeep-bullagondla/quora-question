import sys
import os
import pickle
import mlflow
import pandas as pd
from data_preprocessing import preprocess

# Set MLFLOW experiment
MLFLOW_DIR = "./mlruns"
MLFLOW_EXP_NAME = "quora-question-similarity"
EMB_PATH = "./intermediate_results/embeddings/fasttext_model.pkl"
SCALER_PATH = "./intermediate_results/scaler.pkl"

mlflow.set_tracking_uri(MLFLOW_DIR)
mlflow.set_experiment(MLFLOW_EXP_NAME)

# Load best model
runs = mlflow.search_runs(filter_string="metrics.best_cv_score < 1")
best_run_id = runs.loc[runs['metrics.test_auc_score'].idxmin()]['run_id']
model = mlflow.sklearn.load_model("runs:/" + best_run_id + "/model")


if __name__=="__main__":
    pass
