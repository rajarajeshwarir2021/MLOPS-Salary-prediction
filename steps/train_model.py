import logging
import pandas as pd
from sklearn.base import RegressorMixin
from zenml import step

from src.develop_model import RandomForestRegressorModel

@step
def train_model(X_train: pd.DataFrame, y_train: pd.DataFrame, config_params) -> RegressorMixin:
    """
    Create and Train a Regressor model with the ingested data.
    Args:
        X_train: training dataframe
        y_train: target dataframe
        config_params: configuration parameters object
    """
    try:
        model = None
        if config_params['model_name'] == "RandomForestRegression":
            random_state = config_params["base"]["random_state"]
            n_estimators = config_params["estimators"]["RandomForestRegressor"]["params"]["n_estimators"]
            model = RandomForestRegressorModel().train(X_train, y_train, random_state, n_estimators)
            return model
        else:
            raise ValueError(f"Model {config_params['model_name']} not supported.")
    except Exception as e:
        logging.error(f"Error while training model: {e}")
        raise e


# @step
# def train_regressor_model(config_params: object) -> dict:
#     """
#     Create and Train a Regressor model.
#     Args:
#         config_params: configuration parameters object
#     """
#     logging.info(f"Creating and Training Random Forest Regressor model")
#     try:
#         regressor = RandomForestRegressorModel(config_params)
#         regressor.train()
#         results = regressor.test()
#         regressor.save_parameters()
#         regressor.save_model()
#         return results
#     except Exception as e:
#         logging.error(f"Error while creating and training model: {e}")
#         raise e
#
