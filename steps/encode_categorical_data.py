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
    logging.info(f"Pre-processing data - Encoding categorical data")
    try:
        process_data = EncodeCategoricalData()
        return process_data.pre_process_data(dataframe)
    except Exception as e:
        logging.error(f"Error while pre-processing data - encoding categorical data: {e}")
        raise e