from abc import ABC, abstractmethod
import argparse
import logging
import os
import pandas as pd
from src import label_encoder, one_hot_encoder


class DataPreProcess(ABC):
    """
    Abstract class defining various methods for cleaning and handling data
    """

    @abstractmethod
    def pre_process_data(self, data: pd.DataFrame) -> pd.DataFrame:
        pass


class CleanData(DataPreProcess):
    """
    A class to clean the data by handling the missing values and duplicates
    """
    def pre_process_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Clean data
        Args:
            data: a pandas dataframe
        Returns: a pandas dataframe
        """
        logging.info(f"Cleaning the given dataframe")
        try:
            # Step 1: Remove rows with missing values
            dataframe = data.dropna(how='any', axis=0)
            # Step 2: Remove duplicate rows
            dataframe.drop_duplicates(keep=False, inplace=True)
            # Step 3: Drop the Age column
            dataframe = dataframe.drop('Age', axis=1)
            return dataframe
        except Exception as e:
            logging.error(f"Error while cleaning data: {e}")
            raise e


class EncodeCategoricalData(DataPreProcess):
    """
    A class to encode the categorical data
    """
    def pre_process_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Encode Categorical data using Label or One Hot encoding
        Args:
            data: a pandas dataframe to be encoded
        Returns: a pandas dataframe
        """
        logging.info(f"Encoding categorical columns in the given dataframe")
        try:
            # Step 1: Label Encode rankable Categorical columns
            dataframe = label_encoder.label_encode(data, 'Education Level')
            dataframe = label_encoder.label_encode(dataframe, 'Job Title')

            # Step 2: One Hot Encode non-rankable Categorical Data Column
            dataframe = one_hot_encoder.one_hot_encode(dataframe, 0)

            # Step 3: Convert numpy array to pandas DataFrame
            dataframe = pd.DataFrame(dataframe)
            return dataframe
        except Exception as e:
            logging.error(f"Error while encoding categorical data: {e}")
            raise e


class RefineData(DataPreProcess):
    """
    A class to refine the dataframe
    """
    def pre_process_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Refine the data
        Args:
            data: a pandas dataframe to refine
        Returns: a pandas dataframe
        """
        logging.info(f"Refining the given dataframe")
        try:
            # Step 1: Rename column names
            new_cols = [col.replace(" ", "_") for col in data.columns]
            data.columns = new_cols
            return data
        except Exception as e:
            logging.error(f"Error while refining data: {e}")
            raise e




















class PreProcessData:
    """
    A class for pre-processing a dataframe.
    """
    def __init__(self, dataframe: pd.DataFrame, config):
        """
        Args:
            dataframe: a pandas dataframe to clean and pre-process
            config: configuration object
        """
        self.dataframe = dataframe
        self.interim_data_path = config['preprocess_data_source']['interim_dataset_csv']

    def pre_process_data(self):
        """
        Processing the data from the dataframe.
        Returns: A pandas DataFrame
        """






        return self.dataframe

@step
def preprocess_data(dataframe: pd.DataFrame, config: object) -> pd.DataFrame:
    """
    Preprocess the given dataframe
    Args:
        dataframe: pandas dataframe to process
        config: configuration object
    Returns:
        A pandas Dataframe
    """
    try:
        process_data = PreProcessData(dataframe, config)
        return process_data.pre_process_data()
    except Exception as e:
        logging.error(f"Error while pre-processing data: {e}")
        raise e


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    default_config_path = os.path.join("config", "params.yaml")
    args.add_argument('--config', type=str, default=default_config_path)
    parsed_args = args.parse_args()

    config = read_config.read_config(config_path=parsed_args.config)
    raw_data_path = config['data_source']['data_source_path']
    interim_data_path = config['preprocess_data_source']['interim_dataset_csv']
    dataframe = ingest_data.ingest_data(data_path=raw_data_path)
    processed_dataframe = preprocess_data(dataframe, config)
