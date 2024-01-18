from abc import ABC, abstractmethod
import logging

import joblib


class PredictionService(ABC):
    """
    Abstract class for model prediction
    """
    @abstractmethod
    def predict(self, X_predict, config_params: object):
        """
        Predict on the model
        Args:
            X_predict: pandas dataframe for prediction
            config_params: configuration parameters object
        Returns:
            A pandas dataframe
        """
        pass

class PredictRegressor(PredictionService):
    """
    A class to predict on a Random Forest regression model.
    """
    def predict(self, X_predict, config_params: object) -> float:
        """
        Predict on the model
        Args:
            X_predict: pandas dataframe for prediction
            config_params: configuration parameters object
        Returns:
            A float representing the regression output
        """
        try:
            model_path = config_params['model']['model_path']
            model = joblib.load(model_path)
            prediction = [model.predict(X_predict)][0]
            logging.info(f"Model prediction completed")
            return prediction
        except Exception as e:
            logging.error(f"Error while model prediction: {e}")
            raise e
