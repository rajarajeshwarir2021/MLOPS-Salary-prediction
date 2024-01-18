from zenml import pipeline

from steps.evaluate_model import evaluate_model
from steps.get_dataset import get_dataset
from steps.read_config_params import read_config
from steps.save_artifacts import save_artifacts
from steps.train_model import train_model


@pipeline(enable_cache=False)
def train_pipeline(config_path:str):
    config_params = read_config(config_path)
    X_train, y_train, X_test, y_test = get_dataset(config_params)
    model = train_model(X_train, y_train, config_params)
    metrics = evaluate_model(model, X_test, y_test)
    save_artifacts(model, metrics, config_params)


