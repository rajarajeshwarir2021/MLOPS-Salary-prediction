import joblib
import logging
import os
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder


class ColumnOneHotEncoder():
    """
    A class for One Hot encoding a column in pandas dataframe.
    """
    def __init__(self, dataframe: pd.DataFrame, column: int):
        """
        Args:
            dataframe: pandas dataframe to encode
            column: column number to one hot encode
        """
        self.dataframe = dataframe
        self.column = column
        self.ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [self.column])], remainder='passthrough')

    def encode(self):
        """
        Encode the dataframe column
        Returns:
            An object
        """
        logging.info(f"One Hot Encoding column {self.column}")
        self.dataframe = self.ct.fit_transform(self.dataframe)

        return self.dataframe

    def save_encoder(self):
        """
        Save the One Hot encoder
        """
        logging.info(f"Saving the One Hot Encoder")
        encoder_path = os.path.join("encoder", "OneHotEncoder.joblib")
        try:
            joblib.dump(self.ct, encoder_path, compress=9)
        except Exception as e:
            logging.error(f"Error while saving One Hot encoder: {e}")
            raise e

def one_hot_encode(dataframe: pd.DataFrame, column: int) -> object:
    """
    One hot encode the dataframe column
    Args:
        dataframe: pandas dataframe to encode
        column: column number to one hot encode
    Returns:
        An object
    """
    try:
        encoder = ColumnOneHotEncoder(dataframe, column)
        df = encoder.encode()
        encoder.save_encoder()
        return df
    except Exception as e:
        logging.error(f"Error while One Hot encoding: {e}")
        raise e
