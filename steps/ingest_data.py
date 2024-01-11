import argparse
import logging
import os
import pandas as pd
from steps import read_config


class IngestData:
    """
    A class for ingesting data from the data_path.
    """
    def __init__(self, data_path: str):
        """
        Args:
            data_path: path to the data to be ingested
        """
        self.data_path = data_path

    def get_data(self):
        """
        Ingesting the data from the data_path.
        Returns: A pandas DataFrame
        """
        logging.info(f"Ingesting data from {self.data_path}")
        return pd.read_csv(self.data_path, sep=",", encoding="utf-8")


def ingest_data(data_path: str) -> pd.DataFrame:
    """
    Ingest data from the data_path
    Args:
        data_path: path to the data to ingest
    Returns:
        A pandas Dataframe
    """
    try:
        ingest_data = IngestData(data_path)
        return ingest_data.get_data()
    except Exception as e:
        logging.error(f"Error while ingesting data: {e}")
        raise e


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    default_config_path = os.path.join("config", "params.yaml")
    args.add_argument('--config', type=str, default=default_config_path)
    parsed_args = args.parse_args()

    config = read_config.read_config(config_path=parsed_args.config)
    data_path = config['data_source']['data_source_path']
    dataframe = ingest_data(data_path=data_path)

