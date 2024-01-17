from zenml import pipeline

from steps.clean_data import clean_data
from steps.encode_categorical_data import encode_data
from steps.get_data import get_data
from steps.read_config_params import read_config
from steps.refine_data import refine_data
from steps.save_processed_dataframe import save_dataframe
from steps.save_dataframe_schema import save_schema
from steps.split_dataset import split_data


@pipeline(enable_cache=True)
def data_pipeline(config_path:str):
    config_params = read_config(config_path)
    dataframe = get_data(config_params)
    dataframe = clean_data(dataframe)
    dataframe = refine_data(dataframe)
    save_schema(dataframe, config_params)
    dataframe = encode_data(dataframe)
    save_dataframe(dataframe, config_params)
    split_data(dataframe, config_params)