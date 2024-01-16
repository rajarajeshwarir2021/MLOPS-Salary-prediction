import logging
from zenml import step
from src.read_config import ReadConfig


@step
def read_config(config_path: str) -> object:
    """
    Read the parameters from the given yaml config file
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
