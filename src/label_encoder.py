import joblib
import json
import logging
import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder


class ColumnLabelEncoder():
    """
    A class for Label encoding a column in pandas dataframe.
    """
    def __init__(self, dataframe: pd.DataFrame, column_name: str):
        """
        Args:
            dataframe: pandas dataframe to encode
            column_name: column name to label encode
        """
        self.dataframe = dataframe
        self.column_name = column_name
        self.label_encoder = LabelEncoder()

    def encode(self):
        """
        Encode the dataframe column
        Returns:
            An object
        """
        logging.info(f"Label Encoding column {self.column_name}")
        self.dataframe[self.column_name] = self.label_encoder.fit_transform(self.dataframe[self.column_name])

        return self.dataframe

    def save_encoder(self):
        """
        Save the Label encoder
        """
        logging.info(f"Saving the Label Encoder")
        encoder_path = os.path.join("encoder", f"{self.column_name}_LabelEncoder.joblib")
        try:
            joblib.dump(self.label_encoder, encoder_path, compress=9)
        except Exception as e:
            logging.error(f"Error while saving Label encoder: {e}")
            raise e

def label_encode(dataframe: pd.DataFrame, column_name: str) -> object:
    """
    Label encode the dataframe column
    Args:
        dataframe: pandas dataframe to encode
        column_name: column name to label encode
    Returns:
        An object
    """
    try:
        encoder = ColumnLabelEncoder(dataframe, column_name)
        df = encoder.encode()
        encoder.save_encoder()
        return df
    except Exception as e:
        logging.error(f"Error while Label encoding: {e}")
        raise e
