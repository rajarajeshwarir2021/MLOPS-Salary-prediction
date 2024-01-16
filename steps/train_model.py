import logging
from zenml import step

from src.train_model import RandomForestRegressorModel


@step
def train_regressor_model(config_params: object) -> list:
    """
    Create and Train a Regressor model.
    Args:
        config_params: configuration parameters object
    """
    logging.info(f"Creating and Training Random Forest Regressor model")
    try:
        regressor = RandomForestRegressorModel(config_params)
        regressor.train()
        y_test, y_predict = regressor.test()
        regressor.save_parameters()
        regressor.save_model()
        return y_test, y_predict
    except Exception as e:
        logging.error(f"Error while creating and training the model: {e}")
        raise e

