from zenml import pipeline

from steps.evaluate_model import evaluate_regressor_model
from steps.read_config_params import read_config
from steps.train_model import train_regressor_model


@pipeline(enable_cache=True)
def train_pipeline(config_path:str):
    config_params = read_config(config_path)
    y_result = train_regressor_model(config_params)
    evaluate_regressor_model(y_result, config_params)


