from pipelines.data_pipeline import data_pipeline
from pipelines.train_pipeline import train_pipeline

if __name__ == '__main__':
    # Run the data pipeline
    #data_pipeline(config_path="config/params.yaml")
    # Run the train pipeline
    train_pipeline(config_path="config/params.yaml")

