import logging
import numpy as np
from typing import Tuple
from typing_extensions import Annotated

from src.ingest_data import IngestData


class SplitDataset:
    """
    A class for splitting the x and y from the data in the data_path.
    """
    def __init__(self, data_path: str):
        """
        Args:
            data_path: path to the data to be split
        """
        self.data_path = data_path

    def split_dataset(self) -> Tuple[Annotated[np.ndarray, "X_split"], Annotated[np.ndarray, "y_split"]]:
        """
        Splitting the data from the data_path.
        Returns: A pandas DataFrame
        """
        logging.info(f"Splitting dataset from {self.data_path}")
        try:
            data = IngestData(data_path=self.data_path)
            dataframe = data.ingest_data()
            x_df = dataframe.iloc[:, :-1].values
            y_df = dataframe.iloc[:, -1].values
            return x_df, y_df
        except Exception as e:
            logging.error(f"Error while splitting dataset: {e}")
            raise e
