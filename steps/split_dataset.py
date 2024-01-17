import logging
import pandas as pd
from zenml import step

from src.split_data import SplitDataframe


@step
def split_data(dataframe: pd.DataFrame, config_params: object) -> None:
    """
    Splits the data into train and test set
    Args:
        dataframe: a pandas dataframe
        config_params: config object containing parameters to split the dataset
    """
    logging.info(f"Splitting dataset")
    try:
        split_df = SplitDataframe()
        split_df.split_data(dataframe, config_params)
    except Exception as e:
        logging.error(f"Error while splitting dataset: {e}")
        raise e
