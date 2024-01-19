from zenml import pipeline

from steps.create_dataframe import create_input_dataframe
from steps.predict_data import predict_data
from steps.read_config_params import read_config


@pipeline(enable_cache=True)
def inference_pipeline(config_path:str, user_data):
    config_params = read_config(config_path)
    dataframe = create_input_dataframe(user_data, config_params)
    prediction = predict_data(dataframe, config_params)
    return prediction
