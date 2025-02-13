import os
import sys
import dill
import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
import pickle

from src.exception import CustomException
from src.logger import logging

def save_object(file_path, obj):
    """
    Saves a Python object to a file using dill.

    Args:
        file_path (str): Path where the object should be saved.
        obj: The Python object to be saved.
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
        
        logging.info(f"Object saved successfully at {file_path}")

    except Exception as e:
        logging.error(f"Error saving object: {str(e)}")
        raise CustomException(e, sys)

def evaluate_models(X_train, y_train, X_test, y_test, models, params):
    """
    Evaluates multiple models using GridSearchCV and returns a performance report.

    Args:
        X_train, y_train: Training data and labels.
        X_test, y_test: Test data and labels.
        models (dict): Dictionary of models to evaluate.
        params (dict): Hyperparameter grid for each model.

    Returns:
        dict: Model names as keys and R2 score as values.
    """
    try:
        report = {}

        for model_name, model in models.items():
            logging.info(f"Evaluating model: {model_name}")

            param_grid = params.get(model_name, {})
            gs = GridSearchCV(model, param_grid, cv=3, n_jobs=-1, verbose=1)
            gs.fit(X_train, y_train)

            best_params = gs.best_params_
            model.set_params(**best_params)
            model.fit(X_train, y_train)

            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            report[model_name] = test_model_score

            logging.info(f"{model_name}: Train R2 Score = {train_model_score:.4f}, Test R2 Score = {test_model_score:.4f}")

        return report

    except Exception as e:
        logging.error(f"Error evaluating models: {str(e)}")
        raise CustomException(e, sys)


def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)