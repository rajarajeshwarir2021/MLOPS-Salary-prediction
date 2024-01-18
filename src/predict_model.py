from abc import ABC, abstractmethod
import argparse
import logging
import os
from sklearn.ensemble import RandomForestRegressor

from src.read_config import ReadConfig


class PredictionService(ABC):
    """
    Abstract class for model prediction
    """
    @abstractmethod
    def predict(self, X_predict, config_params: object):
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
    def predict(self, X_predict, config_params: object) -> float:
        """
        Predict on the model
        Args:
            X_predict: pandas dataframe for prediction
            config_params: configuration parameters object
        Returns:
            A float representing the regression output
        """
        try:

            logging.info(f"Model prediction completed")

            return
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
