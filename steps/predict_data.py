import logging
import pandas as pd
from zenml import step

from src.create_dataframe import FormulateInput
from src.validate_input import ValidateInput


@step
def create_input_dataframe(data: dict, config_params: object) -> pd.DataFrame:
    """
    Create the input dataframe expected by the model.
    Args:
        data: the input data
        config_params: configuration parameters object
    Returns:
        A pandas DataFrame
    """
    logging.info(f"Formulating input dataframe")
    try:
        if ValidateInput.validate(data, config_params):
            dataframe = FormulateInput.formulate(data, config_params)
            return dataframe
    except Exception as e:
        logging.error(f"Error while formulating input dataframe: {e}")
        raise e


