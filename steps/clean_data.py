import logging
import pandas as pd
from zenml import step

from src.preprocess_data import CleanData


@step
def clean_data(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the given dataframe
    Args:
        dataframe: pandas dataframe to clean
    Returns:
        A pandas Dataframe
    """
    logging.info(f"Pre-processing data - cleaning data")
    try:
        process_data = CleanData()
        return process_data.pre_process_data(dataframe)
    except Exception as e:
        logging.error(f"Error while pre-processing data - cleaning data: {e}")
        raise e
