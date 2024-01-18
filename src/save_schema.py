from abc import ABC, abstractmethod
import logging
import os
import pandas as pd


class SaveSchema(ABC):
    """
    Abstract class to save/ store pandas dataframe schema
    """

    @abstractmethod
    def save_schema(self, data: pd.DataFrame, config_params: object):
        pass


class SaveSchemaJSON(SaveSchema):
    """
    A class to save the dataframe schema in JSON format
    """
    def save_schema(self, data: pd.DataFrame, config_params: object):
        """
        Save schema
        Args:
            data: a pandas dataframe
            config_params: configuration parameters object
        """
        logging.info(f"Saving given dataframe schema in JSON format")
        schema_path = config_params['preprocess_data_source']['dataset_schema_json']
        os.makedirs(schema_path, exist_ok=True)
        schema_path = os.path.join(schema_path, "dataset_schema.json")

        try:
            # Save the min max values of the dataframe
            overview = data.describe()
            overview.loc[["min", "max"]].to_json(schema_path)
        except Exception as e:
            logging.error(f"Error while saving dataframe schema in JSON format: {e}")
            raise e
