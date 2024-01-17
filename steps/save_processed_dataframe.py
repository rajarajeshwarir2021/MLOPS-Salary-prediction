import logging
import pandas as pd
from zenml import step

from src.save_dataframe import SaveDataframeCSV


@step
def save_dataframe(dataframe: pd.DataFrame, config_params: object):
    """
    Save the processed dataframe
    Args:
        dataframe: pandas dataframe to refine
        config_params: config object containing path to processed data folder
    """
    logging.info(f"Saving processed dataframe")
    try:
        save_df = SaveDataframeCSV()
        save_df.save_dataframe(dataframe, config_params)
    except Exception as e:
        logging.error(f"Error while saving processed dataframe: {e}")
        raise e