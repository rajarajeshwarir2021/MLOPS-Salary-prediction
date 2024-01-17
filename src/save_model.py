from abc import ABC, abstractmethod
import joblib
import logging
import os
from sklearn.base import RegressorMixin


class SaveModel(ABC):
    """
    Abstract class to save/ store trained model.
    """

    @abstractmethod
    def save_model(self, model: RegressorMixin, config_params: object):
        pass


class SaveModelJoblib(SaveModel):
    """
    A class to save the model in joblib format.
    """
    def save_model(self, model: RegressorMixin, config_params: object):
        """
        Save data
        Args:
            model: a machine learning model
            config_params: configuration parameters object
        """
        try:
            model_dir_path = config_params['model_dir']
            os.makedirs(model_dir_path, exist_ok=True)
            model_file_path = os.path.join(model_dir_path, "model.joblib")
            joblib.dump(model, model_file_path)
            logging.info(f"Model saved")
        except Exception as e:
            logging.error(f"Error while saving model: {e}")
            raise e




