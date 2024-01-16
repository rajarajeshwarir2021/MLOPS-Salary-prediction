from abc import ABC, abstractmethod
import joblib
import logging
import os
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from typing_extensions import Annotated


class EncodeCategory(ABC):
    """
    Abstract class for encoding column in a dataframe
    """
    def __init__(self, data: pd.DataFrame, column:Annotated[str, int]):
        self.dataframe = data
        self.column = column
        self.encoder = None

    @abstractmethod
    def encode(self) -> pd.DataFrame:
        pass

    @abstractmethod
    def save_encoder(self):
        pass


class LabelEncodeColumn(EncodeCategory):
    """
    A class to Label encode a column in a dataframe
    """
    def __init__(self, data: pd.DataFrame, column:Annotated[str, int]):
        self.dataframe = data
        self.column = column
        self.encoder = LabelEncoder()

    def encode(self) -> pd.DataFrame:
        """
        Encode column
        Args:
        Returns: a pandas dataframe
        """
        logging.info(f"Label Encoding column {self.column}")
        try:
            self.dataframe[self.column] = self.encoder.fit_transform(self.dataframe[self.column])
            return self.dataframe
        except Exception as e:
            logging.error(f"Error while Label encoding: {e}")
            raise e

    def save_encoder(self):
        """
        Save the Label encoder
        """
        logging.info(f"Saving the Label Encoder")
        os.makedirs("encoder", exist_ok=True)
        encoder_path = os.path.join("encoder", f"{self.column}_LabelEncoder.joblib")
        try:
            joblib.dump(self.encoder, encoder_path, compress=9)
        except Exception as e:
            logging.error(f"Error while saving Label encoder: {e}")
            raise e


class OneHotEncodeColumn(EncodeCategory):
    """
    A class to One Hot encode a column in a dataframe
    """
    def __init__(self, data: pd.DataFrame, column:Annotated[str, int]):
        self.dataframe = data
        self.column = column
        self.encoder = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [self.column])], remainder='passthrough')

    def encode(self) -> pd.DataFrame:
        """
        Encode column
        Returns: a pandas dataframe
        """
        logging.info(f"One Hot Encoding column {self.column}")
        try:
            self.dataframe = self.encoder.fit_transform(self.dataframe)
            return self.dataframe
        except Exception as e:
            logging.error(f"Error while One Hot encoding: {e}")
            raise e

    def save_encoder(self):
        """
        Save the Label encoder
        """
        logging.info(f"Saving the One Hot Encoder")
        os.makedirs("encoder", exist_ok=True)
        encoder_path = os.path.join("encoder", "OneHotEncoder.joblib")
        try:
            joblib.dump(self.encoder, encoder_path, compress=9)
        except Exception as e:
            logging.error(f"Error while saving One Hot encoder: {e}")
            raise e

