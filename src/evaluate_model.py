from abc import ABC, abstractmethod
import logging
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


class Evaluate(ABC):
    """
    Abstract class to evaluate a model.
    """

    @abstractmethod
    def calculate_metric(self, y_true: pd.DataFrame, y_pred: pd.DataFrame):
        """
        Calculates the metrics of the model.
        Args:
            y_true: true values
            y_pred: predicted values
        Returns:
            A float representing the metrics of the model
        """
        pass


class MSE(Evaluate):
    """
    A class to evaluate the mean squared error of a model.
    """
    def calculate_metric(self, y_true: pd.DataFrame, y_pred: pd.DataFrame):
        """
        Calculates the mean squared error of the model.
        Args:
            y_true: true values
            y_pred: predicted values
        Returns:
            A float representing the mean squared error of the model
        """
        try:
            logging.info("Calculating Mean Squared Error")
            mse = round(mean_squared_error(y_true, y_pred), 2)
            logging.info(f"MSE: {mse}")
            return mse
        except Exception as e:
            logging.error(f"Error while calculating MSE: {e}")


class RMSE(Evaluate):
    """
    A class to evaluate the root mean squared error of a model.
    """
    def calculate_metric(self, y_true: pd.DataFrame, y_pred: pd.DataFrame):
        """
        Calculates the root mean squared error of the model.
        Args:
            y_true: true values
            y_pred: predicted values
        Returns:
            A float representing the root mean squared error of the model
        """
        try:
            logging.info("Calculating Root Mean Squared Error")
            rmse = round(np.sqrt(mean_squared_error(y_true, y_pred)), 2)
            logging.info(f"RMSE: {rmse}")
            return rmse
        except Exception as e:
            logging.error(f"Error while calculating RMSE: {e}")


class MAE(Evaluate):
    """
    A class to evaluate the mean absolute error of a model.
    """
    def calculate_metric(self, y_true: pd.DataFrame, y_pred: pd.DataFrame):
        """
        Calculates the mean absolute error of the model.
        Args:
            y_true: true values
            y_pred: predicted values
        Returns:
            A float representing the mean absolute error of the model
        """
        try:
            logging.info("Calculating Mean Absolute Error")
            mae = round(mean_absolute_error(y_true, y_pred), 2)
            logging.info(f"MAE: {mae}")
            return mae
        except Exception as e:
            logging.error(f"Error while calculating MAE: {e}")


class R2(Evaluate):
    """
    A class to evaluate the R2 score of a model.
    """
    def calculate_metric(self, y_true: pd.DataFrame, y_pred: pd.DataFrame):
        """
        Calculates the R2 score of the model.
        Args:
            y_true: true values
            y_pred: predicted values
        Returns:
            A float representing the R2 score of the model
        """
        try:
            logging.info("Calculating R2 score")
            r2 = round(r2_score(y_true, y_pred), 2)
            logging.info(f"R2: {r2}")
            return r2
        except Exception as e:
            logging.error(f"Error while calculating R2 score: {e}")



