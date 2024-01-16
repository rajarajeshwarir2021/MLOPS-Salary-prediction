import argparse
import logging
import os
import pandas as pd
from src.read_config import ReadConfig


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

    def ingest_data(self):
        """
        Ingesting the data from the data_path.
        Returns: A pandas DataFrame
        """
        logging.info(f"Ingesting data from {self.data_path}")
        return pd.read_csv(self.data_path, sep=",", encoding="utf-8")


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    default_config_path = os.path.join("config", "params.yaml")
    args.add_argument('--config', type=str, default=default_config_path)
    parsed_args = args.parse_args()

    config = ReadConfig(config_path=parsed_args.config)
    params = config.read_params()
    data_path = params['data_source']['data_source_path']
    df = IngestData(data_path=data_path)
    dataframe = df.ingest_data()
    print(dataframe)

