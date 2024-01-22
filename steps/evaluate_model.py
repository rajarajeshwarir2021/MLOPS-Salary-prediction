import logging

import mlflow
import numpy as np
from sklearn.base import RegressorMixin
from zenml import step
from zenml.client import Client

from src.evaluate_model import MSE, RMSE, MAE, R2

experiment_tracker = Client().active_stack.experiment_tracker


@step(experiment_tracker=experiment_tracker.name)
def evaluate_model(model: RegressorMixin, X_test: np.ndarray, y_test: np.ndarray) -> list:
    """
    Evaluate a Regressor model.
    Args:
        model: trained Machine Learning model
        X_test: test dataframe
        y_test:  test target dataframe
    Returns:

    """
    logging.info(f"Evaluating model")
    try:
        prediction = model.predict(X_test)
        mse = MSE().calculate_metric(y_test, prediction)
        mlflow.log_metric("MSE", mse)
        rmse = RMSE().calculate_metric(y_test, prediction)
        mlflow.log_metric("RMSE", rmse)
        mae = MAE().calculate_metric(y_test, prediction)
        mlflow.log_metric("MAE", mae)
        r2 = R2().calculate_metric(y_test, prediction)
        mlflow.log_metric("R2", r2)
        return [mse, rmse, mae, r2]
    except Exception as e:
        logging.error(f"Error while evaluating model: {e}")
        raise e

















# @step
# def evaluate_regressor_model(result: dict, config_params: object):
#     """
#     Evaluate a Regressor model.
#     Args:
#         actual: pandas dataframe of actual values
#         predict: pandas dataframe of predicted values
#         config_params: configuration parameters object
#     """
#     logging.info(f"Evaluating Random Forest Regressor model")
#     try:
#         eval_model = EvaluateRegressorModel(result['actual'], result['predict'])
#         eval_model.calculate_metrics()
#         eval_model.save_metrics(config_params)
#     except Exception as e:
#         logging.error(f"Error while evaluating model: {e}")
#         raise e

