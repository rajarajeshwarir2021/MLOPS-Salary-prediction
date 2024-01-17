import logging
import pandas as pd
from sklearn.base import RegressorMixin
from typing import Tuple
from typing_extensions import Annotated
from zenml import step

from src.evaluate_model import MSE, RMSE, MAE, R2


@step
def evaluate_model(model: RegressorMixin, X_test: pd.DataFrame, y_test: pd.DataFrame) -> Tuple[Annotated[float, "mse"], Annotated[float, "rmse"], Annotated[float, "mae"], Annotated[float, "r2"]]:
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
        rmse = RMSE().calculate_metric(y_test, prediction)
        mae = MAE().calculate_metric(y_test, prediction)
        r2 = R2().calculate_metric(y_test, prediction)
        return mse, rmse, mae, r2
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

