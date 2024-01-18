import argparse
import os
from abc import ABC, abstractmethod
import logging

import joblib
import numpy as np

from src.read_config import ReadConfig


class PredictionService(ABC):
    """
    Abstract class for model prediction
    """
    @abstractmethod
    def predict(self, X_predict: np.ndarray, config_params: object) -> float:
        """
        Predict on the model
        Args:
            X_predict: pandas dataframe for prediction
            config_params: configuration parameters object
        Returns:
            A pandas dataframe
        """
        pass

class PredictRegressor(PredictionService):
    """
    A class to predict on a Random Forest regression model.
    """
    def predict(self, X_predict: np.ndarray, config_params: object) -> float:
        """
        Predict on the model
        Args:
            X_predict: pandas dataframe for prediction
            config_params: configuration parameters object
        Returns:
            A float representing the regression output
        """
        try:
            model_path = config_params["model"]["model_path"]
            reg_model = joblib.load(model_path)
            prediction = [reg_model.predict(X_predict)][0][0]
            print(prediction)
            logging.info(f"Model prediction completed")
            return prediction
        except Exception as e:
            logging.error(f"Error while model prediction: {e}")
            raise e


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    default_config_path = os.path.join("config", "params.yaml")
    args.add_argument('--config', type=str, default=default_config_path)
    parsed_args = args.parse_args()

    config = ReadConfig(config_path=parsed_args.config)
    params = config.read_params()
    input = [[ 0.0, 1.0, 0.0, 157.0, 10.0]]
    data = PredictRegressor().predict(input, params)