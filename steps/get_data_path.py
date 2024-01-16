import argparse
import logging
import os
from zenml import step
from src import read_config


class GetDataPath:
    """
    A class for fetching data path from the yaml config object.
    """
    def __init__(self, config: object):
        """
        Args:
            config: yaml configuration object
        """
        self.config = config

    def get_data_path(self):
        """
        Fetching the data path from the config object.
        Returns: A string representing the data path
        """
        logging.info(f"Fetching data path")
        return self.config['data_source']['data_source_path']

@step
def get_data_path(config: object) -> str:
    """
    Get data path from the config object
    Args:
        config: configuration object
    Returns:
        A string
    """
    try:
        get_path = GetDataPath(config)
        return get_path.get_data_path()
    except Exception as e:
        logging.error(f"Error while fetching the data path: {e}")
        raise e


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    default_config_path = os.path.join("config", "params.yaml")
    args.add_argument('--config', type=str, default=default_config_path)
    parsed_args = args.parse_args()

    config = read_config.read_config(config_path=parsed_args.config)
    data_path = get_data_path(config=config)

