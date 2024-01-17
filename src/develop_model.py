from abc import ABC, abstractmethod
import argparse
import logging
import os
from sklearn.ensemble import RandomForestRegressor

from src.read_config import ReadConfig


class Model(ABC):
    """
    Abstract class for Machine Learning models
    """
    @abstractmethod
    def train(self, X_train, y_train):
        """
        Train the model
        Args:
            X_train: training dataframe
            y_train: target dataframe
        Returns:
        """
        pass

class RandomForestRegressorModel(Model):
    """
    A class to train a Random Forest regression model.
    """
    def train(self, X_train, y_train, n_estimators=5, random_state=None):
        """
        Train the model with train dataset.
        Args:
            X_train: training dataframe
            y_train: target dataframe
            n_estimators: the number of estimators
            random_state: the random state seed to reproduce the results
        Returns: A regressor model
        """
        logging.info(f"Training the model with train dataset")
        try:
            regressor = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
            regressor.fit(X_train, y_train)
            logging.info(f"Model training completed")
            return regressor
        except Exception as e:
            logging.error(f"Error while training model: {e}")
            raise e


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    default_config_path = os.path.join("config", "params.yaml")
    args.add_argument('--config', type=str, default=default_config_path)
    parsed_args = args.parse_args()

    config = ReadConfig(config_path=parsed_args.config)
    params = config.read_params()
    regressor = RandomForestRegressorModel(params)
    regressor.train()








