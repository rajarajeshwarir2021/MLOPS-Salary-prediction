from abc import ABC, abstractmethod
import json
import logging


class SaveParameters(ABC):
    """
    Abstract class to save/ store model parameters.
    """

    @abstractmethod
    def save_parameters(self, config_params: object):
        pass


class SaveParametersJSON(SaveParameters):
    """
    A class to save the model parameters in json format.
    """
    def save_parameters(self, config_params: object):
        """
        Save model paramters.
        Args:
            config_params: configuration parameters object
        """
        try:
            params_file_path = config_params["reports"]["params"]
            if config_params['model_name'] == "RandomForestRegression":
                n_estimators = config_params["estimators"]["RandomForestRegressor"]["params"]["n_estimators"]
                params = {
                    "n_estimators": n_estimators
                }
                with open(params_file_path, "w") as f:
                    json.dump(params, f, indent=4)
            logging.info(f"Model parameters saved")
        except Exception as e:
            logging.error(f"Error while saving model parameters: {e}")
            raise e




