import json
from abc import ABC, abstractmethod
import logging
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


class EvaluateModel(ABC):
    """
    Abstract class for evaluating a model.
    """
    def __init__(self, actual: pd.DataFrame, predict: pd.DataFrame):
        """
        Args:
            actual: pandas dataframe of actual values
            predict: pandas dataframe of predicted values
        """
        self.actual = actual
        self.predict = predict

    @abstractmethod
    def calculate_mse(self):
        pass

    @abstractmethod
    def calculate_rmse(self):
        pass

    @abstractmethod
    def calculate_mae(self):
        pass

    @abstractmethod
    def calculate_r2_score(self):
        pass


class EvaluateRegressorModel():
    """
    A class for evaluating a regressor model.
    """
    def __init__(self, actual: pd.DataFrame, predict: pd.DataFrame):
        """
        Args:
            actual: pandas dataframe of actual values
            predict: pandas dataframe of predicted values
        """
        self.actual = actual
        self.predict = predict
        self.mse = None
        self.rmse = None
        self.mae = None
        self.r2_score = None

    def calculate_metrics(self):
        """
        Calculate the metrics.
        """
        logging.info(f"Calculating the metrics")
        self.calculate_mse()
        self.calculate_rmse()
        self.calculate_mae()
        self.calculate_r2_score()
        self.print_metrics()

    def save_metrics(self, config_params: object):
        """
        Save the metrics.
        Args:
            config_params: configuration parameters object
        """
        logging.info(f"Saving the metrics")
        metrics_file_path = config_params["reports"]["scores"]
        metrics = {
            "MSE": self.mse,
            "RMSE": self.rmse,
            "MAE": self.mae,
            "R2": self.r2_score
        }
        with open(metrics_file_path, "w") as f:
            json.dump(metrics, f, indent=4)

    def print_metrics(self):
        """
        Print the metrics.
        """
        print(f"MSE: {self.mse}\nRMSE: {self.rmse}\nMAE: {self.mae}\nR2: {self.r2_score}")

    def calculate_mse(self):
        """
        Calculates the mean squared error of the model.
        Returns:
            A float representing the mean squared error
        """
        self.mse = round(mean_squared_error(self.actual, self.predict), 2)
        return self.mse

    def calculate_rmse(self):
        """
        Calculates the root mean squared error of the model.
        Returns:
            A float representing the root mean squared error
        """
        self.rmse = round(np.sqrt(self.mse),2)
        return self.rmse

    def calculate_mae(self):
        """
        Calculates the mean absolute error of the model.
        Returns:
            A float representing the mean absolute error
        """
        self.mae = round(mean_absolute_error(self.actual, self.predict), 2)
        return self.mae

    def calculate_r2_score(self):
        """
        Calculates the R2 score of the model.
        Returns:
            A float representing the R2 score
        """
        self.r2_score = round(r2_score(self.actual, self.predict), 2)
        return self.r2_score
