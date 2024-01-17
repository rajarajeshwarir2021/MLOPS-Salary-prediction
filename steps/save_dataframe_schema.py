import logging
import pandas as pd
from zenml import step

from src.save_schema import SaveSchemaJSON


@step
def save_schema(dataframe: pd.DataFrame, config_params: object):
    """
    Save the schema of the given dataframe
    Args:
        dataframe: pandas dataframe to refine
        config_params: config object containing path to schema folder
    """
    logging.info(f"Saving dataframe schema")
    try:
        save_sch = SaveSchemaJSON()
        save_sch.save_schema(dataframe, config_params)
    except Exception as e:
        logging.error(f"Error while saving dataframe schema: {e}")
        raise e