from src.read_config import read_config
from steps.get_data_path import get_data_path
from steps.ingest_data import ingest_data
from steps.preprocess_data import preprocess_data
from steps.split_data import split_data
from steps.train_and_evaluate_model import train_and_evaluate_model
from zenml import pipeline

@pipeline(enable_cache=True)
def train_pipeline(config_path:str):
    config = read_config(config_path)
    data_path = get_data_path(config)
    df = ingest_data(data_path)
    preprocess_data(df, config)
    split_data(config_path)
    train_and_evaluate_model(config_path)


