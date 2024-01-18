import logging
import pandas as pd
from zenml import step

from src.predict_model import PredictRegressor


@step
def predict_data(data: pd.DataFrame, config_params: object) -> float:
    """
    Model data prediction for the given user input.
    Args:
        data: the input data
        config_params: configuration parameters object
    Returns:
        A pandas DataFrame
    """
    logging.info(f"Predicting on the given input dataframe")
    try:
        prediction = PredictRegressor().predict(data, config_params)
        return prediction
    except Exception as e:
        logging.error(f"Error while predicting on the given input dataframe: {e}")
        raise e


