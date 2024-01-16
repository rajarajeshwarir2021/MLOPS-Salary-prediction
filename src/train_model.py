import argparse
import json
from abc import ABC, abstractmethod
import joblib
import logging
import os
from sklearn.ensemble import RandomForestRegressor

from src.evaluate_model import EvaluateRegressorModel
from src.ingest_data import IngestData
from src.read_config import ReadConfig


class Model(ABC):
    """
    Abstract class for training and testing a model.
    """
    def __init__(self, config_params: object):
        self.config_params = config_params

    @abstractmethod
    def get_data(self):
        pass

    @abstractmethod
    def get_dataset(self):
        pass

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def test(self):
        pass

    @abstractmethod
    def save_model(self):
        pass


class RandomForestRegressorModel(Model):
    """
    A class to train and test a Random Forest regression model.
    """
    def __init__(self, config_params: object):
        """
        Args:
            config_params: configuration parameters object
        """
        self.config_params = config_params

        # Model parameters
        self.random_state = self.config_params["base"]["random_state"]
        self.n_estimators = self.config_params["estimators"]["RandomForestRegressor"]["params"]["n_estimators"]

        # Create the model
        self.model = RandomForestRegressor(n_estimators=self.n_estimators, random_state=self.random_state)

    def train(self):
        """
        Train the model with train dataset.
        """
        logging.info(f"Training the model with train dataset")
        x_train, y_train = self.get_train_dataset()
        try:
            self.model.fit(x_train, y_train)
        except Exception as e:
            logging.error(f"Error while training model: {e}")
            raise e

    def test(self) -> dict:
        """
        Test the model with test dataset.
        """
        logging.info(f"Testing the model with test dataset")
        x_test, y_test = self.get_test_dataset()
        try:
            y_predict = self.model.predict(x_test)
            return {'actual': y_test, 'predict': y_predict}
        except Exception as e:
            logging.error(f"Error while training model: {e}")
            raise e

    def save_model(self):
        """
        Save the model.
        """
        logging.info(f"Saving the model")
        model_path = self.config_params["model_dir"]
        os.makedirs(model_path, exist_ok=True)
        model_file_path = os.path.join(model_path, "model.joblib")
        joblib.dump(self.model, model_file_path)

    def save_parameters(self):
        """
        Save the parameters.
        """
        logging.info(f"Saving the parameters")
        params_file_path = self.config_params["reports"]["params"]
        params = {
            "n_estimators": self.n_estimators
        }
        with open(params_file_path, "w") as f:
            json.dump(params, f, indent=4)

    def get_train_dataset(self):
        """
        Get train dataset.
        Returns: a tuple of pandas dataframe
        """
        train_data_path = self.config_params['processed_data_source']['train_data_path']
        return self.get_dataset(train_data_path)

    def get_test_dataset(self):
        """
        Get test dataset.
        Returns: a tuple of pandas dataframe
        """
        test_data_path = self.config_params['processed_data_source']['test_data_path']
        return self.get_dataset(test_data_path)

    @staticmethod
    def get_dataset(data_path):
        """
        Get the dataset as x_df and y_df.
        Returns: a tuple of two dataframes
        """
        dataframe = RandomForestRegressorModel.get_data(data_path)
        x_df = dataframe.iloc[:, :-1].values
        y_df = dataframe.iloc[:, -1].values
        return x_df, y_df

    @staticmethod
    def get_data(data_path):
        """
        Get the data.
        Returns: a pandas dataframe
        """
        data = IngestData(data_path=data_path)
        return data.ingest_data()


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    default_config_path = os.path.join("config", "params.yaml")
    args.add_argument('--config', type=str, default=default_config_path)
    parsed_args = args.parse_args()

    config = ReadConfig(config_path=parsed_args.config)
    params = config.read_params()
    regressor = RandomForestRegressorModel(params)
    regressor.train()
    y_test, y_predict = regressor.test()
    regressor.save_model()
    regressor.save_parameters()
    print(y_test)
    print(y_predict)
    eval_model = EvaluateRegressorModel(y_test, y_predict)
    eval_model.calculate_metrics()
    eval_model.save_metrics(params)








