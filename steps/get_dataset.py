import logging
import pandas as pd
from typing import Tuple
from typing_extensions import Annotated
from zenml import step

from src.ingest_data import IngestData


@step
def get_dataset(config_params: object) -> Tuple[Annotated[pd.DataFrame, "X_train"], Annotated[pd.DataFrame, "X_test"], Annotated[pd.DataFrame, "y_train"], Annotated[pd.DataFrame, "y_test"]]:
    """
    Get the train and test splits for model training and evaluation
    Args:
        config_params: configuration parameters object
    Returns:
        Tuple of pandas dataframes
    """
    logging.info(f"Ingesting train and test datasets")
    try:
        train_data_path = config_params['processed_data_source']['train_data_path']
        test_data_path = config_params['processed_data_source']['test_data_path']
        train_df = IngestData(train_data_path)
        test_df = IngestData(test_data_path)
        X_train = train_df.iloc[:, :-1].values
        y_train = train_df.iloc[:, -1].values
        X_test = test_df.iloc[:, :-1].values
        y_test = test_df.iloc[:, -1].values
        return X_train, y_train, X_test, y_test
    except Exception as e:
        logging.error(f"Error while ingesting train and test datasets: {e}")
        raise e


