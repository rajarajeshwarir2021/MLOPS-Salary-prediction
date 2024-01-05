import os
import yaml
import logging
import pandas as pd
from zenml import steps


class IngestData:
    """
    A class for ingesting data from the data_path.
    """
    def __init__(self, data_path: str):
        """
        Args:
            data_path: path to the data
        """
        self.data_path = data_path

    def get_data(self):
        """
        Ingesting the data from the data_path.
        Returns: A pandas DataFrame
        """
        logging.info(f"Ingesting data from {self.data_path}")
        return pd.read_csv(self.data_path, sep=",", encoding="utf-8")

@steps
def ingest_data(data_path: str) -> pd.DataFrame:
    """
    Ingest data from the data_path
    Args:
        data_path: path to the data
    Returns:
        A pandas Dataframe
    """
    try:
        ingest_data = IngestData(data_path)
        return ingest_data.get_data()
    except Exception as e:
        logging.error(f"Error while ingesting data: {e}")
        raise e