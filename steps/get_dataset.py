import logging
import numpy as np
from typing import Tuple
from typing_extensions import Annotated
from zenml import step

from src.get_dataset_split import SplitDataset



@step
def get_dataset(config_params: object) -> Tuple[Annotated[np.ndarray, "X_train"], Annotated[np.ndarray, "y_train"], Annotated[np.ndarray, "X_test"], Annotated[np.ndarray, "y_test"]]:
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
        train_split = SplitDataset(train_data_path)
        X_train, y_train = train_split.split_dataset()
        test_split = SplitDataset(test_data_path)
        X_test, y_test = test_split.split_dataset()
        return X_train, y_train, X_test, y_test
    except Exception as e:
        logging.error(f"Error while ingesting train and test datasets: {e}")
        raise e


