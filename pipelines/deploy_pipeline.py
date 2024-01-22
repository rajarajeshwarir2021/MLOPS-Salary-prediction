import numpy as np
import pandas as pd
from zenml import pipeline, step
from zenml.config import DockerSettings
from zenml.constants import  DEFAULT_SERVICE_START_STOP_TIMEOUT
from zenml.integrations.constants import MLFLOW
from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import (MLFlowModelDeployer)
from zenml.integrations.mlflow.services import MLFlowDeploymentService
from zenml.integrations.mlflow.steps import mlflow_model_deployer_step

from steps.evaluate_model import evaluate_model
from steps.get_dataset import get_dataset
from steps.read_config_params import read_config
from steps.save_artifacts import save_artifacts
from steps.train_model import train_model

docker_settings = DockerSettings(required_integrations=[MLFLOW])

@step
def deploy_trigger(config_params: object, metrics: list) -> bool:
    """
    The trigger to deploy the model
    Args:
        config_params: configuration parameters object
        metrics: list of regressor metrics
    Returns:
        A bool
    """
    #min_accuracy = config_params["deploy"]["min_accuracy"]
    min_accuracy = 0.95
    r2 = metrics[3]

    if r2 >= min_accuracy:
        return True
    else:
        return False


@pipeline(enable_cache=True, settings={"docker": docker_settings})
def continuous_deploy_pipeline(config_path: object, workers: int = 1,
                               timeout: int = DEFAULT_SERVICE_START_STOP_TIMEOUT,
                               ):
    """
    Pipeline to deploy continuously
    Args:
        config_path: path to the configuration file
        workers: number of workers to deploy
        timeout: timeout in seconds
    """
    config_params = read_config(config_path)
    print(config_params)
    X_train, y_train, X_test, y_test = get_dataset(config_params)
    model = train_model(X_train, y_train, config_params)
    metrics = evaluate_model(model, X_test, y_test)
    save_artifacts(model, metrics, config_params)
    deployment_decision = deploy_trigger(config_params, metrics)
    mlflow_model_deployer_step(model=model, deploy_decision=deployment_decision, workers=workers, timeout=timeout)

