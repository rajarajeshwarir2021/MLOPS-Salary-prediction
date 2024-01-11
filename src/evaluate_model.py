import logging
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


class EvaluateModel():
    """
    A class for evaluating the model.
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

    def calculate_mse(self):
        """
        Calculates the mean squared error of the model
        Returns:
            A float representing the mean squared error
        """
        logging.info(f"Calculating the Mean Squared Error")
        self.mse = round(mean_squared_error(self.actual, self.predict), 2)
        return self.mse

    def calculate_rmse(self):
        """
        Calculates the root mean squared error of the model
        Returns:
            A float representing the root mean squared error
        """
        logging.info(f"Calculating the Root Mean Squared Error")
        self.rmse = round(np.sqrt(self.mse),2)
        return self.rmse

    def calculate_mae(self):
        """
        Calculates the mean absolute error of the model
        Returns:
            A float representing the mean absolute error
        """
        logging.info(f"Calculating the Mean Absolute Error")
        self.mae = round(mean_absolute_error(self.actual, self.predict), 2)
        return self.mae

    def calculate_r2_score(self):
        """
        Calculates the R2 score of the model
        Returns:
            A float representing the R2 score
        """
        logging.info(f"Calculating the R2 score")
        self.r2_score = round(r2_score(self.actual, self.predict), 2)
        return self.r2_score
def evaluate_metrics(actual: pd.DataFrame, predict: pd.DataFrame) -> object:
    """
    Evaluate the metrics.
    Args:
        actual: pandas dataframe of actual values
        predict: pandas dataframe of predicted values
    Returns:
        An object
    """
    eval_metrics = {}
    try:
        evaluate_model = EvaluateModel(actual, predict)
        eval_metrics["MSE"] = evaluate_model.calculate_mse()
        eval_metrics["RMSE"] = evaluate_model.calculate_rmse()
        eval_metrics["MAE"] = evaluate_model.calculate_mae()
        eval_metrics["R2"] = evaluate_model.calculate_r2_score()
        return eval_metrics
    except Exception as e:
        logging.error(f"Error while Evaluating the model: {e}")
        raise e
