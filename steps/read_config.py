import argparse
import logging
import os
import yaml
from zenml import steps

class ReadConfig():
    """
    A class for reading the parameters from the given yaml config file.
    """
    def __init__(self, config_path: str):
        """
        Args:
            config_path: path to the config file
        """
        self.config_path = config_path

    def read_params(self):
        """
        Read the parameters from the given config file
        Returns:
            An object
        """
        logging.info(f"Reading parameters from {self.config_path}")
        with open(self.config_path, 'r') as f:
            params = yaml.safe_load(f)
        return params

@steps
def read_config(config_path: str) -> object:
    """
    Read the parameters from the given config file
    Args:
        config_path: path to the config file
    Returns:
        An object
    """
    try:
        read_config = ReadConfig(config_path)
        return read_config.read_params()
    except Exception as e:
        logging.error(f"Error while reading parameters: {e}")
        raise e


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    default_config_path = os.path.join("config", "params.yaml")
    args.add_argument('--config', type=str, default=default_config_path)
    parsed_args = args.parse_args()

    config = read_config(config_path=parsed_args.config)