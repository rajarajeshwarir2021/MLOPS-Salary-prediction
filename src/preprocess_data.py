from abc import ABC, abstractmethod
import argparse
import logging
import os
import pandas as pd
from src.encode_category import OneHotEncodeColumn, LabelEncodeColumn
from src.ingest_data import IngestData
from src.read_config import ReadConfig
from src.save_dataframe import SaveDataframeCSV
from src.save_schema import SaveSchemaJSON


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
            # Step 1: Label Encode rankable Categorical columns and save the encoder
            encoder = LabelEncodeColumn(data, 'Education_Level')
            dataframe = encoder.encode()
            encoder.save_encoder()
            encoder = LabelEncodeColumn(dataframe, 'Job_Title')
            dataframe = encoder.encode()
            encoder.save_encoder()

            # Step 2: One Hot Encode non-rankable Categorical Data Column and save the encoder
            encoder = OneHotEncodeColumn(dataframe, 0)
            dataframe = encoder.encode()
            encoder.save_encoder()

            # Step 3: Convert numpy array to pandas DataFrame
            dataframe = pd.DataFrame(dataframe)

            return dataframe
        except Exception as e:
            logging.error(f"Error while encoding categorical data: {e}")
            raise e


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    default_config_path = os.path.join("config", "params.yaml")
    args.add_argument('--config', type=str, default=default_config_path)
    parsed_args = args.parse_args()

    config = ReadConfig(config_path=parsed_args.config)
    params = config.read_params()
    data_path = params['data_source']['data_source_path']
    df = IngestData(data_path=data_path)
    dataframe = df.ingest_data()
    clean_data = CleanData()
    dataframe = clean_data.pre_process_data(dataframe)
    refine_data = RefineData()
    dataframe = refine_data.pre_process_data(dataframe)
    save_sch = SaveSchemaJSON()
    save_sch.save_schema(dataframe, params)
    encode_data = EncodeCategoricalData()
    dataframe = encode_data.pre_process_data(dataframe)
    save_df = SaveDataframeCSV()
    save_df.save_dataframe(dataframe, params)
    print(dataframe)


