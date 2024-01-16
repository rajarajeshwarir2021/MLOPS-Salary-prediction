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


class Model:
    """
    A class for creating a model and training it with the train set and testing it with the test set.
    """

    def __init__(self, config_path: str):

        self.y_predict = None
        self.evaluate_metrics = None


        self.metrics_file = self.config["reports"]["scores"]



    def evaluate(self):
        """
        Evaluate the model.
        """
        logging.info(f"Evaluating the model")
        self.evaluate_metrics = evaluate_model.evaluate_metrics(self.y_test, self.y_predict)
        print(f"ElasticNet model (n_estimators={self.n_estimators}):")
        print(f"MSE: {self.evaluate_metrics['MSE']}")
        print(f"RMSE: {self.evaluate_metrics['RMSE']}")
        print(f"MAE: {self.evaluate_metrics['MAE']}")
        print(f"R2: {self.evaluate_metrics['R2']}")



    def save_metrics(self):
        """
        Save the metrics.
        """
        logging.info(f"Saving the metrics")
        with open(self.metrics_file, "w") as f:
            metrics = {
                "MSE" : self.evaluate_metrics['MSE'],
                "RMSE" : self.evaluate_metrics['RMSE'],
                "MAE" : self.evaluate_metrics['MAE'],
                "R2" : self.evaluate_metrics['R2']
            }
            json.dump(metrics, f, indent=4)