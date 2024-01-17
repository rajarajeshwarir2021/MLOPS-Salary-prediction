from abc import ABC, abstractmethod
import json
import logging
from typing import Tuple


class SaveMetrics(ABC):
    """
    Abstract class to save/ store evaluated model metrics.
    """

    @abstractmethod
    def save_metrics(self, metrics: Tuple, config_params: object):
        pass


class SaveMetricsJSON(SaveMetrics):
    """
    A class to save the model metrics in json format.
    """
    def save_metrics(self, metrics: Tuple, config_params: object):
        """
        Save metrics.
        Args:
            metrics: evaluated model metrics
            config_params: configuration parameters object
        """
        try:
            metrics_file_path = config_params["reports"]["scores"]
            metrics = {
                "MSE": metrics[0],
                "RMSE": metrics[1],
                "MAE": metrics[2],
                "R2": metrics[3]
            }
            with open(metrics_file_path, "w") as f:
                json.dump(metrics, f, indent=4)
            logging.info(f"Model metrics saved")
        except Exception as e:
            logging.error(f"Error while saving model metrics: {e}")
            raise e




