import logging
import pandas as pd
from zenml import step

from src.preprocess_data import EncodeCategoricalData


@step
def encode_data(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Encode the given dataframe
    Args:
        dataframe: pandas dataframe to encode
    Returns:
        A pandas Dataframe
    """
    try:
        process_data = EncodeCategoricalData()
        return process_data.pre_process_data(dataframe)
    except Exception as e:
        logging.error(f"Error while pre-processing data: {e}")
        raise e