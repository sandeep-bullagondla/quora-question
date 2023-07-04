import os
import pickle
import numpy as np
import pandas as pd
from typing import Any, Dict, List

from fuzzywuzzy import fuzz
from gensim.models import FastText

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.model_selection import GridSearchCV
from sklearn import metrics

import mlflow
from prefect import task, flow

from data_preprocessing import preprocess

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot


@task
def load_data(path: str, unwanted_cols: List) -> pd.DataFrame:
    data = pd.read_csv(path)
    data.dropna(inplace=True)
    data.drop(unwanted_cols, axis=1, inplace=True)
    return data


def count_common_words(q1: str, q2: str) -> int:
    w1 = set(map(lambda word: word.lower().strip(), q1.split()))
    w2 = set(map(lambda word: word.lower().strip(), q2.split()))
    return len(w1 & w2)


def count_total_words(q1: str, q2: str) -> int:
    w1 = set(map(lambda word: word.lower().strip(), q1.split()))
    w2 = set(map(lambda word: word.lower().strip(), q2.split()))
    return (len(w1) + len(w2))


def fetch_fuzzy_features(q1: str, q2: str, prefix="") -> pd.Series:
    fuzzy_features = [0.0]*4
    fuzzy_features[0] = fuzz.QRatio(q1, q2)             # fuzz_ratio
    fuzzy_features[1] = fuzz.partial_ratio(q1, q2)      # fuzz_partial_ratio
    fuzzy_features[2] = fuzz.token_sort_ratio(q1, q2)   # token_sort_ratio
    fuzzy_features[3] = fuzz.token_set_ratio(q1, q2)    # token_set_ratio
    # Feature names
    names = ["fuzz_ratio", "fuzz_partial_ratio", "token_sort_ratio", "token_set_ratio"]
    if prefix:
        names = ["_".join((prefix, n)) for n in names]
    return pd.Series(fuzzy_features, index=names)


def get_document_vector(doc: list, wv) -> np.array:
    if not doc:
        return wv.__getitem__("").reshape(1, -1)
    return np.mean(wv.__getitem__(doc), axis=0).reshape(1, -1)


@task
def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
    # Preprocess data
    qid1 = data.loc[:, ["qid1", "question1"]].drop_duplicates()\
               .rename(columns = {"qid1": "qid", "question1": "question"})
    qid2 = data.loc[:, ["qid2", "question2"]].drop_duplicates()\
               .rename(columns = {"qid2": "qid", "question2": "question"})
    all_questions = pd.concat([qid1, qid2], ignore_index=True).drop_duplicates().set_index("qid").squeeze()
    cleaned_questions = all_questions.apply(lambda q: preprocess(q, 'lemma'))
    return cleaned_questions


@task
def train_embeddings(data: pd.Series, save_path: str):
    tokenised_sentences = data.str.split().tolist()
    model = FastText(tokenised_sentences,
                     vector_size=100,
                     window=5,
                     min_count=1,
                     workers=4,
                     sg=1)
    with open(save_path, "wb") as f:
        pickle.dump(model, f)
    return model


def compute_distances(data: pd.DataFrame, cleaned_questions: pd.Series, emb_path: str):
    with open(emb_path, "rb") as f:
        ft_model = pickle.load(f)
    
    tokenised_data = cleaned_questions.str.split()
    encoded_questions = tokenised_data.apply(lambda x: get_document_vector(x, ft_model.wv))
    encoded_q1 = data["qid1"].map(encoded_questions)
    encoded_q2 = data["qid2"].map(encoded_questions)
    
    distances = {
        "euclid_dist": np.array([metrics.pairwise_distances(q1, q2, metric="euclidean").flatten()[0]
                                for q1, q2 in zip(encoded_q1, encoded_q2)]),
        "manhattan_dist": np.array([metrics.pairwise_distances(q1, q2, metric="manhattan").flatten()[0]
                                for q1, q2 in zip(encoded_q1, encoded_q2)]),
        "cosine_dist": np.array([metrics.pairwise_distances(q1, q2, metric="cosine").flatten()[0]
                                for q1, q2 in zip(encoded_q1, encoded_q2)])
    }
    out_df = pd.DataFrame.from_records(distances, index=data.index)
    return out_df


@task
def feature_engineering(data: pd.DataFrame, cleaned_questions: pd.Series, emb_path: str, train: bool) -> pd.DataFrame:
    features_df = data.copy(deep=True)

    features_df["cleaned_question1"] = features_df["qid1"].map(cleaned_questions)
    features_df["cleaned_question2"] = features_df["qid2"].map(cleaned_questions)

    # Question length
    features_df['q1_len'] = features_df['cleaned_question1'].str.len()
    features_df['q2_len'] = features_df['cleaned_question2'].str.len()

    # Number of words
    features_df['q1_num_words'] = features_df['cleaned_question1'].apply(lambda q: len(q.split()))
    features_df['q2_num_words'] = features_df['cleaned_question2'].apply(lambda q: len(q.split()))

    # Number of common words
    features_df['num_common_words'] = features_df.apply(lambda row: count_common_words(row['cleaned_question1'], row['cleaned_question2']), axis=1)

    # Total number of words
    features_df['total_words'] = features_df.apply(lambda row: count_total_words(row['cleaned_question1'], row['cleaned_question2']), axis=1)

    # Proportion of common words
    features_df['proportion_common_words'] = features_df['num_common_words'] / (features_df['total_words'] + 0.01)

    # Similar type of question
    features_df["similar_question_type"] = features_df.apply(lambda row: row["question1"].split()[0] == row["question2"].split()[0],
                                                       axis=1).astype(float)

    # Fuzzy features
    raw_fuzzy_features = features_df.apply(lambda row: fetch_fuzzy_features(row["question1"], row["question2"], "raw"), axis=1)
    cleaned_fuzzy_features = features_df.apply(lambda row: fetch_fuzzy_features(row["cleaned_question1"], row["cleaned_question2"], "cleaned"), axis=1)
    features_df = pd.concat([features_df, raw_fuzzy_features, cleaned_fuzzy_features], axis=1)

    # Compute distance in embedded space
    distance_features = compute_distances(features_df, cleaned_questions, emb_path)
    features_df = pd.concat([features_df, distance_features], axis=1)
    
    # Remove unwanted columns
    features_df.drop(["qid1", "qid2", "question1", "question2", "cleaned_question1", "cleaned_question2"], axis=1, inplace=True)

    return features_df


@task
def get_classes(target_data: pd.Series) -> List[str]:
    return list(target_data.unique())


@task
def get_scaler(data: pd.DataFrame) -> Any:
    # scaling the numerical features
    scaler = StandardScaler()
    scaler.fit(data)
    return scaler


@task
def rescale_data(data: pd.DataFrame, scaler: Any) -> pd.DataFrame:    
    # scaling the numerical features
    # column names are (annoyingly) lost after Scaling
    # (i.e. the dataframe is converted to a numpy ndarray)
    data_rescaled = pd.DataFrame(scaler.transform(data), 
                                columns = data.columns, 
                                index = data.index)
    return data_rescaled


@task
def split_data(input_: pd.DataFrame, output_: pd.Series, test_data_ratio: float, random_state: int) -> Dict[str, Any]:
    X_tr, X_te, y_tr, y_te = train_test_split(input_, output_, test_size=test_data_ratio, random_state=random_state)
    return {'X_TRAIN': X_tr, 'Y_TRAIN': y_tr, 'X_TEST': X_te, 'Y_TEST': y_te}


def eval_test_metrics(actual, pred):
    out = {
        "test_accuracy": metrics.accuracy_score(actual, pred),
        "test_precision": metrics.precision_score(actual, pred),
        "test_recall": metrics.recall_score(actual, pred),
        "test_f1_score": metrics.f1_score(actual, pred),
        "test_auc_score": metrics.roc_auc_score(actual, pred)
    }
    return out


@task
def find_best_model(train_test_dict: dict, estimator: Any, parameters: List, run_name: str) -> Any:
    # Enabling automatic MLflow logging for scikit-learn runs
    mlflow.sklearn.autolog(max_tuning_runs=None)
    
    X_train, y_train = train_test_dict["X_TRAIN"], train_test_dict["Y_TRAIN"]
    X_test, y_test = train_test_dict["X_TEST"], train_test_dict["Y_TEST"]
    with mlflow.start_run(run_name=run_name):
        clf = GridSearchCV(
            estimator=estimator, 
            param_grid=parameters, 
            scoring='roc_auc',
            cv=5,
            n_jobs=4,
            return_train_score=True,
            verbose=1
        )
        clf.fit(X_train, y_train)
        
        y_test_pred = clf.predict(X_test)
        mlflow.log_metrics(eval_test_metrics(y_test, y_test_pred))
        
        # Disabling autologging
        mlflow.sklearn.autolog(disable=True)
        
        return clf


# Workflow
@flow
def main(path: str, force_rerun: bool):

    FORCE_RERUN = force_rerun
    MLFLOW_DIR = "./mlruns"
    MLFLOW_EXP_NAME = "quora-question-similarity"
    RANDOM_STATE = 12181006
    EMB_PATH = "./intermediate_results/embeddings/fasttext_model.pkl"
    SCALER_PATH = "./intermediate_results/scaler.pkl"
    TRAIN_TEST_PATH = "./intermediate_results/train_test_dict.pkl"
    
    
    mlflow.set_tracking_uri(MLFLOW_DIR)
    mlflow.set_experiment(MLFLOW_EXP_NAME)
    
    # Define Parameters
    TARGET_COL = 'is_duplicate'
    UNWANTED_COLS = ['id']
    TEST_DATA_RATIO = 0.3
    DATA_PATH = path

    if FORCE_RERUN:
        # Load the Data
        dataframe = load_data(path=DATA_PATH, unwanted_cols=UNWANTED_COLS)

        # Identify Target Variable
        target_data = dataframe[TARGET_COL]
        input_data = dataframe.drop([TARGET_COL], axis=1)
        
        # Split the Data into Train and Test
        train_test_dict = split_data(input_=input_data, output_=target_data,
                                     test_data_ratio=TEST_DATA_RATIO, random_state=RANDOM_STATE)

        # Data preprocessing
        cleaned_train_data = preprocess_data(train_test_dict['X_TRAIN'])
        cleaned_test_data = preprocess_data(train_test_dict['X_TEST'])

        # Train embeddings
        train_embeddings(cleaned_train_data, EMB_PATH)

        # Feature engineering
        train_test_dict['X_TRAIN'] = feature_engineering(train_test_dict['X_TRAIN'], cleaned_train_data, emb_path=EMB_PATH, train=True)
        train_test_dict['X_TEST'] = feature_engineering(train_test_dict['X_TEST'], cleaned_test_data, emb_path=EMB_PATH, train=False)
        with open(TRAIN_TEST_PATH, "wb") as f:
            pickle.dump(train_test_dict, f)

        # Rescaling Train and Test Data
        scaler = get_scaler(train_test_dict['X_TRAIN'])
        train_test_dict['X_TRAIN'] = rescale_data(data=train_test_dict['X_TRAIN'], scaler=scaler)
        train_test_dict['X_TEST'] = rescale_data(data=train_test_dict['X_TEST'], scaler=scaler)

        with open(SCALER_PATH, "wb") as f:
            pickle.dump(scaler, f)

        with open(TRAIN_TEST_PATH, "wb") as f:
            pickle.dump(train_test_dict, f)
    else:
        with open(TRAIN_TEST_PATH, "rb") as f:
            train_test_dict = pickle.load(f)

    # Model Training
    # Logistic Regression
    RUN_NAME = "logistic_reg"
    ESTIMATOR = LogisticRegression(class_weight="balanced", random_state=RANDOM_STATE, max_iter=1000, solver="liblinear")
    HYPERPARAMETERS = [{'penalty': ['l1', 'l2'],
                        'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}]
    classifier = find_best_model(train_test_dict, ESTIMATOR, HYPERPARAMETERS, RUN_NAME)

    # Decision tree
    RUN_NAME = "decision_tree"
    ESTIMATOR = DecisionTreeClassifier(class_weight="balanced", random_state=RANDOM_STATE)
    HYPERPARAMETERS = [{'max_features': ['sqrt', 'log2'],
                        'ccp_alpha': [0.1, .01, .001],
                        'max_depth' : [5, 6, 7, 8, 9],
                        'criterion' :['gini', 'entropy']}]
    classifier = find_best_model(train_test_dict, ESTIMATOR, HYPERPARAMETERS, RUN_NAME)

    # Random Forest
    RUN_NAME = "random_forest"
    ESTIMATOR = RandomForestClassifier(class_weight="balanced", random_state=RANDOM_STATE)
    HYPERPARAMETERS = [{'n_estimators': [200, 500],
                        'max_features': ['sqrt', 'log2'],
                        'max_depth' : [4,5,6,7,8],
                        'criterion' :['gini', 'entropy']}]
    classifier = find_best_model(train_test_dict, ESTIMATOR, HYPERPARAMETERS, RUN_NAME)

    # XGBoost
    RUN_NAME = "xgboost"
    ESTIMATOR = XGBClassifier(objective= 'binary:logistic', nthread=4, seed=RANDOM_STATE)
    HYPERPARAMETERS = [{'max_depth': range (2, 10, 1),
                        'n_estimators': range(60, 220, 400),
                        'learning_rate': [0.1, 0.01, 0.05]}]
    classifier = find_best_model(train_test_dict, ESTIMATOR, HYPERPARAMETERS, RUN_NAME)
    
    
# Run the main function
main(path='./data/train.csv', force_rerun=False)
