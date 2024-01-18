import logging
import pandas as pd
from zenml import step


@step
def validate_input(config_params: object) -> bool:
    """
    Validate the input data.
    Args:
        config_params: configuration parameters object
    Returns:
        A bool
    """
    logging.info(f"Ingesting raw datasource")
    data_path = config_params['data_source']['data_source_path']
    try:
        ingest_data = IngestData(data_path)
        return ingest_data.ingest_data()
    except Exception as e:
        logging.error(f"Error while ingesting raw datasource: {e}")
        raise e


