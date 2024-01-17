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
    logging.info(f"Reading configuration parameters")
    try:
        read_config = ReadConfig(config_path)
        return read_config.read_params()
    except Exception as e:
        logging.error(f"Error while reading configuration parameters: {e}")
        raise e
