import argparse
import logging
import os
import pandas as pd
from src import label_encoder, one_hot_encoder
from steps import read_config, ingest_data


class PreprocessData:
    """
    A class for pre-processing data from the data_path.
    """
    def __init__(self, dataframe: pd.DataFrame):
        """
        Args:
            dataframe: a pandas dataframe to pre-process
        """
        self.dataframe = dataframe

    def process_data(self):
        """
        Processing the data from the dataframe.
        Returns: A pandas DataFrame
        """
        logging.info(f"Processing data from the given dataframe")

        # Step 1: Remove rows with missing values
        self.dataframe = self.dataframe.dropna(how='any', axis=0)

        # Step 2: Remove duplicate rows
        self.dataframe.drop_duplicates(keep=False, inplace=True)

        # Step 3: Rename column names
        new_cols = [col.replace(" ", "_") for col in self.dataframe.columns]
        self.dataframe.columns = new_cols

        # Step 4: Drop the Age column
        self.dataframe = self.dataframe.drop('Age', axis=1)

        # Step 5: Save the min max schema of the dataframe
        overview = self.dataframe.describe()
        overview.loc[["min", "max"]].to_json("schema_data.json")

        # Step 6: Label Encode rankable Categorical columns
        self.dataframe = label_encoder.label_encode(self.dataframe, 'Education_Level')
        self.dataframe = label_encoder.label_encode(self.dataframe, 'Job_Title')

        # Step 7: One Hot Encode non-rankable Categorical Data Column
        self.dataframe = one_hot_encoder.one_hot_encode(self.dataframe, 0)

        # Step 8: Convert numpy array to pandas DataFrame
        self.dataframe = pd.DataFrame(self.dataframe)

        return self.dataframe


def preprocess_data(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the given dataframe
    Args:
        dataframe: a pandas dataframe to pre-process
    Returns:
        A pandas Dataframe
    """
    try:
        preprocess_data = PreprocessData(dataframe)
        return preprocess_data.process_data()
    except Exception as e:
        logging.error(f"Error while pre-processing data: {e}")
        raise e


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    default_config_path = os.path.join("config", "params.yaml")
    args.add_argument('--config', type=str, default=default_config_path)
    parsed_args = args.parse_args()
    config = read_config.read_config(config_path=parsed_args.config)
    data_path = config['data_source']['data_source_path']
    interim_data_path = config['preprocess_data_source']['interim_dataset_csv']
    dataframe = ingest_data.ingest_data(data_path)
    processed_dataframe = preprocess_data(dataframe)
    processed_dataframe.to_csv(interim_data_path, sep=",", encoding="utf-8", index=False)