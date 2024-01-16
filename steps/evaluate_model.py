import logging
from zenml import step

from src.evaluate_model import EvaluateRegressorModel


@step
def evaluate_regressor_model(actual, predict, config_params: object):
    """
    Evaluate a Regressor model.
    Args:
        actual: pandas dataframe of actual values
        predict: pandas dataframe of predicted values
        config_params: configuration parameters object
    """
    logging.info(f"Evaluating Random Forest Regressor model")
    try:
        eval_model = EvaluateRegressorModel(actual, predict)
        eval_model.calculate_metrics()
        eval_model.save_metrics(config_params)
    except Exception as e:
        logging.error(f"Error while evaluating model: {e}")
        raise e

