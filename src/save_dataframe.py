from abc import ABC, abstractmethod
import logging
import pandas as pd


class SaveDataframe(ABC):
    """
    Abstract class to save/ store pandas dataframe
    """

    @abstractmethod
    def save_dataframe(self, data: pd.DataFrame, config_params: object):
        pass


class SaveDataframeCSV(SaveDataframe):
    """
    A class to save the dataframe in CSV format
    """
    def save_dataframe(self, data: pd.DataFrame, config_params: object):
        """
        Save data
        Args:
            data: a pandas dataframe
            config_params: configuration parameters object
        """
        logging.info(f"Saving given dataframe in CSV format")
        interim_path = config_params['preprocess_data_source']['interim_dataset_csv']

        try:
            # Save the dataframe
            data.to_csv(interim_path, sep=",", encoding="utf-8", index=False)
        except Exception as e:
            logging.error(f"Error while saving dataframe: {e}")
            raise e
