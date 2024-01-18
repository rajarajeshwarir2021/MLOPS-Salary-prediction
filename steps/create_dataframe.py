import logging
import numpy as np
from zenml import step

from src.create_dataframe import FormulateFormInput
from src.validate_input import ValidateUserInput


@step
def create_input_dataframe(data: dict, config_params: object) -> np.ndarray:
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
        validate_input = ValidateUserInput(data, config_params)
        if validate_input.validate():
            dataframe = FormulateFormInput().formulate(data, config_params)
            return dataframe
    except Exception as e:
        logging.error(f"Error while formulating input dataframe: {e}")
        raise e


