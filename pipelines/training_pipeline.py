from steps.read_config import read_config
from steps.ingest_data import ingest_data
from steps.preprocess_data import preprocess_data
from steps.split_data import split_data
from steps.train_and_evaluate_model import train_and_evaluate_model
from zenml import pipelines

@pipelines
def train_pipeline(config_path:str):
    config = read_config(config_path)
    raw_data_path = config['data_source']['data_source_path']
    df = ingest_data(raw_data_path)
    interim_data_path = config['preprocess_data_source']['interim_dataset_csv']
    processed_df = preprocess_data(df, interim_data_path)
    split_data(config_path)
    train_and_evaluate_model(config_path)


