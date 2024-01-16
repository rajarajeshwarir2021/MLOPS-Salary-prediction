from zenml import pipeline

from steps.read_config_params import read_config
from steps.train_and_evaluate_model import train_and_evaluate_model

@pipeline(enable_cache=True)
def train_pipeline(config_path:str):
    config = read_config(config_path)
    #train_and_evaluate_model(config_path)


