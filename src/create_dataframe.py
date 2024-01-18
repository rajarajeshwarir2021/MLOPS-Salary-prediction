from abc import ABC, abstractmethod
import joblib
import logging
import numpy as np
import os
import pandas as pd


def create_dataframe(data):
    """
    Create a data frame from the data
    """
    data = np.array([list(data.values())])
    headers = ['Gender', 'Education_Level', 'Job_Title', 'Years_of_Experience']
    df = pd.DataFrame(data, columns=headers)
    return df

def load_encoder(encoder_path):
    """
    Load the encoder
    """
    encoder = joblib.load(encoder_path)
    return encoder

def label_encode(dataframe, config_params):
    """
    Label Encode the categorical columns of the dataframe
    """
    for item in ['Education_Level', 'Job_Title']:
        encoder_path = config_params["preprocess_data_source"]['encoder_path']
        encoder_file_path = os.path.join(encoder_path, f"{item}_LabelEncoder.joblib")
        label_encoder = load_encoder(encoder_file_path)
        dataframe[item] = label_encoder.transform(dataframe[item])
    return dataframe

def one_hot_encode(dataframe, config_params):
    """
    One Hot Encode the categorical columns of the dataframe
    """
    encoder_path = config_params["preprocess_data_source"]['encoder_path']
    encoder_file_path = os.path.join(encoder_path, "OneHotEncoder.joblib")
    one_hot_encoder = load_encoder(encoder_file_path)
    dataframe = one_hot_encoder.transform(dataframe)
    return dataframe

class FormulateDataframe(ABC):
    """
    Abstract class for formulating a dataframe.
    """
    @abstractmethod
    def formulate(self, data, config_params) -> pd.DataFrame:
        """
        Formulate the input dataframe.
        Args:
            data: data to formulate
            config_params: configuration parameters object
        Returns:
            A pandas dataframe
        """
        pass

class FormulateInput(FormulateDataframe):
    """
    A class for formulating the input dataframe.
    """
    def formulate(self, data, config_params) -> pd.DataFrame:
        """
        Formulate the input dataframe.
        Args:
            data: data to formulate
            config_params: configuration parameters object
        Returns:
            A pandas dataframe
        """
        try:
            dataframe = create_dataframe(data)
            dataframe = label_encode(dataframe, config_params)
            dataframe = one_hot_encode(dataframe, config_params)
            data = dataframe.values.tolist()[0]
            data = [list(map(float, data))]
            logging.info(f"Formulated input dataframe")
            return data
        except Exception as e:
            logging.error(f"Error while formulating input dataframe: {e}")
            raise e

