from abc import ABC, abstractmethod
import argparse
import logging
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from src.ingest_data import IngestData
from src.preprocess_data import CleanData, RefineData, EncodeCategoricalData
from src.read_config import ReadConfig
from src.save_dataframe import SaveDataframeCSV
from src.save_schema import SaveSchemaJSON


class SplitDataset(ABC):
    """
    Abstract class for splitting data into train and test set.
    """

    @abstractmethod
    def split_data(self, data: pd.DataFrame, config_params: object):
        pass


class SplitDataframe(SplitDataset):
    """
    A class for splitting dataframe into train and test set.
    """
    def split_data(self, data: pd.DataFrame, config_params: object):
        """
        Split data
        Args:
            data: a pandas dataframe
            config_params: configuration parameters object
        """
        logging.info("Splitting data into train and test set")

        random_state = config_params["base"]["random_state"]
        train_data_path = config_params['processed_data_source']['train_data_path']
        test_data_path = config_params['processed_data_source']['test_data_path']
        split_ratio = config_params['processed_data_source']['test_size']

        try:
            train, test = train_test_split(data, test_size=split_ratio, random_state=random_state)
            train.to_csv(train_data_path, sep=",", encoding="utf-8", index=False)
            test.to_csv(test_data_path, sep=",", encoding="utf-8", index=False)
        except Exception as e:
            logging.error(f"Error while splitting data: {e}")
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
    split_df = SplitDataframe()
    split_df.split_data(dataframe, params)
