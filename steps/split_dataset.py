import argparse
import logging
import os
from sklearn.model_selection import train_test_split
from zenml import step
from steps import get_data
from src import read_config


class SplitData:
    """
    A class for splitting data into train and test set.
    """
    def __init__(self, config_path: str):
        """
        Args:
            config_path: path to the params.yaml config file
        """
        self.config = read_config.read_config(config_path=config_path)

        self.random_state = self.config["base"]["random_state"]
        self.interim_data_path = self.config['preprocess_data_source']['interim_dataset_csv']
        self.train_data_path = self.config['processed_data_source']['train_data_path']
        self.test_data_path = self.config['processed_data_source']['test_data_path']
        self.split_ratio = self.config['processed_data_source']['test_size']

        self.dataframe = get_data.ingest_data(self.interim_data_path)

    def split_data(self):
        """
        Split the data from the dataframe into train and test set based on the split ratio.
        """
        logging.info(f"Splitting data into train and test set")
        train, test = train_test_split(self.dataframe, test_size=self.split_ratio, random_state=self.random_state)
        train.to_csv(self.train_data_path, sep=",", encoding="utf-8", index=False)
        test.to_csv(self.test_data_path, sep=",", encoding="utf-8", index=False)

@step
def split_data(config_path: str) -> None:
    """
    Splits the data into train and test set
    Args:
        config_path: path to the config file
    """
    try:
        data_split = SplitData(config_path)
        data_split.split_data()
    except Exception as e:
        logging.error(f"Error while splitting data: {e}")
        raise e


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    default_config_path = os.path.join("config", "params.yaml")
    args.add_argument('--config', type=str, default=default_config_path)
    parsed_args = args.parse_args()
    split_data(config_path=parsed_args.config)

