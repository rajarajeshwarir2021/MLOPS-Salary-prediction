import logging
from sklearn.base import RegressorMixin
from typing import Tuple
from zenml import step

from src.save_metrics import SaveMetricsJSON
from src.save_model import SaveModelJoblib
from src.save_parameters import SaveParametersJSON


@step
def save_artifacts(model: RegressorMixin, metrics: Tuple, config_params: object):
    """
    Save the model artifacts.
    Args:
        model: trained Machine Learning model
        metrics: tuple of regressor model metrics
        config_params: configuration parameters
    """
    logging.info(f"Saving model artifacts")
    try:
        SaveModelJoblib().save_model(model, config_params)
        SaveParametersJSON().save_parameters(config_params)
        SaveMetricsJSON().save_metrics(metrics, config_params)
    except Exception as e:
        logging.error(f"Error while saving model artifacts: {e}")
        raise e