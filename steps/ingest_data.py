import argparse
import logging
import os
import pandas as pd
from steps import read_config


class IngestData:
    """
    A class for ingesting data from the data_path.
    """
    def __init__(self, config_path: str):
        """
        Args:
            config_path: path to the params.yaml config file
        """
        self.config = read_config.read_config(config_path=config_path)
        self.data_path = self.config['data_source']['data_source_path']

    def get_data(self):
        """
        Ingesting the data from the data_path.
        Returns: A pandas DataFrame
        """
        logging.info(f"Ingesting data from {self.data_path}")
        return pd.read_csv(self.data_path, sep=",", encoding="utf-8")


def ingest_data(config_path: str) -> pd.DataFrame:
    """
    Ingest data from the data_path
    Args:
        config_path: path to the config file
    Returns:
        A pandas Dataframe
    """
    try:
        ingest_data = IngestData(config_path)
        return ingest_data.get_data()
    except Exception as e:
        logging.error(f"Error while ingesting data: {e}")
        raise e


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    default_config_path = os.path.join("config", "params.yaml")
    args.add_argument('--config', type=str, default=default_config_path)
    parsed_args = args.parse_args()
    dataframe = ingest_data(config_path=parsed_args.config)

