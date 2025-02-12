import sys
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression

from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from src.exception import CustomException
from src.logger import logging

from src.utils import save_object, evaluate_models  # ✅ Fixed function name

@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_training(self, train_arr, test_arr):
        try:
            logging.info("Entered the model training method or component")

            # ✅ Extract input features and target
            X_train, y_train, X_test, y_test = (
                train_arr[:, :-1],
                train_arr[:, -1],
                test_arr[:, :-1],
                test_arr[:, -1],
            )

            # ✅ Define models
            models = {
                "LinearRegression": LinearRegression(),
                "DecisionTreeRegressor": DecisionTreeRegressor(),
                "RandomForestRegressor": RandomForestRegressor(),
                "AdaBoostRegressor": AdaBoostRegressor(),
                "GradientBoostingRegressor": GradientBoostingRegressor(),
                "KNeighborsRegressor": KNeighborsRegressor(),
                "CatBoostRegressor": CatBoostRegressor(verbose=False),
            }

            # ✅ Call evaluate_model (Fixed function name)
            model_report: dict = evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models)

            # ✅ Get the best model name & score correctly
            best_model_name = max(model_report, key=model_report.get)  # Get key with max value
            best_model_score = model_report[best_model_name]

            # ✅ Check if the best model has a good score
            if best_model_score < 0.6:
                logging.warning(f"Best model {best_model_name} has a low score: {best_model_score}")
                raise CustomException("Model score is less than 0.6", sys)

            logging.info(f"Best model: {best_model_name} with score: {best_model_score}")

            best_model = models[best_model_name]

            # ✅ Save the best model
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            # ✅ Predict and calculate R² score
            predicted = best_model.predict(X_test)
            r2_square = r2_score(y_test, predicted)

            return r2_square

        except Exception as e:
            raise CustomException(e, sys)
