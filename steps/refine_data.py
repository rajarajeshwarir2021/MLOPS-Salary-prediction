import logging
import pandas as pd
from zenml import step

from src.preprocess_data import RefineData


@step
def refine_data(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Refine the given dataframe
    Args:
        dataframe: pandas dataframe to refine
    Returns:
        A pandas Dataframe
    """
    logging.info(f"Pre-processing data - refining data")
    try:
        process_data = RefineData()
        return process_data.pre_process_data(dataframe)
    except Exception as e:
        logging.error(f"Error while pre-processing data - refining data: {e}")
        raise e