import logging
import pandas as pd
from zenml import step

from src.ingest_data import IngestData


@step
def get_data(config_params: object) -> pd.DataFrame:
    """
    Get the raw datasource from the config object
    Args:
        config_params: config object containing path to raw datasource
    Returns:
        A pandas Dataframe
    """
    logging.info(f"Ingesting raw datasource")
    data_path = config_params['data_source']['data_source_path']
    try:
        ingest_data = IngestData(data_path)
        return ingest_data.ingest_data()
    except Exception as e:
        logging.error(f"Error while ingesting raw datasource: {e}")
        raise e


