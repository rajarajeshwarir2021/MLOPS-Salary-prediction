import argparse
import joblib
import json
import logging
import os
from sklearn.ensemble import RandomForestRegressor
from zenml import steps
from steps import read_config, ingest_data
from src import evaluate_model


class Model:
    """
    A class for creating a model and training it with the train set and testing it with the test set.
    """

    def __init__(self, config_path: str):
        """
        Args:
            config_path: path to the params.yaml config file
        """
        self.config = read_config.read_config(config_path=config_path)

        self.random_state = self.config["base"]["random_state"]
        self.target = self.config["base"]["target_col"]

        self.train_data_path = self.config['processed_data_source']['train_data_path']
        self.test_data_path = self.config['processed_data_source']['test_data_path']
        self.model_path = self.config["model_dir"]

        self.n_estimators = self.config["estimators"]["RandomForestRegressor"]["params"]["n_estimators"]

        self.train_dataframe = ingest_data.ingest_data(self.train_data_path)
        self.test_dataframe = ingest_data.ingest_data(self.test_data_path)

        # Formulate the train dataset
        self.x_train = self.train_dataframe.iloc[:, :-1].values
        self.y_train = self.train_dataframe.iloc[:, -1].values

        # Formulate the test dataset
        self.x_test = self.test_dataframe.iloc[:, :-1].values
        self.y_test = self.test_dataframe.iloc[:, -1].values

        # Create the model
        self.model = RandomForestRegressor(n_estimators=self.n_estimators, random_state=self.random_state)

        self.y_predict = None
        self.evaluate_metrics = None

        self.params_file = self.config["reports"]["params"]
        self.metrics_file = self.config["reports"]["scores"]
    def train(self):
        """
        Train the model with training dataset.
        """
        logging.info(f"Training the model with train dataset")
        self.model.fit(self.x_train, self.y_train)

    def test(self):
        """
        Test the model with test dataset.
        """
        logging.info(f"Testing the model with test dataset")
        self.y_predict = self.model.predict(self.x_test)

    def evaluate(self):
        """
        Evaluate the model.
        """
        logging.info(f"Evaluating the model")
        self.evaluate_metrics = evaluate_model.evaluate_metrics(self.y_test, self.y_predict)
        print(f"ElasticNet model (n_estimators={self.n_estimators}):")
        print(f"MSE: {self.evaluate_metrics['MSE']}")
        print(f"RMSE: {self.evaluate_metrics['RMSE']}")
        print(f"MAE: {self.evaluate_metrics['MAE']}")
        print(f"R2: {self.evaluate_metrics['R2']}")

    def save_parameters(self):
        """
        Save the parameters.
        """
        logging.info(f"Saving the parameters")
        with open(self.params_file, "w") as f:
            params = {
                "n_estimators": self.n_estimators
            }
            json.dump(params, f, indent=4)

    def save_metrics(self):
        """
        Save the metrics.
        """
        logging.info(f"Saving the metrics")
        with open(self.metrics_file, "w") as f:
            metrics = {
                "MSE" : self.evaluate_metrics['MSE'],
                "RMSE" : self.evaluate_metrics['RMSE'],
                "MAE" : self.evaluate_metrics['MAE'],
                "R2" : self.evaluate_metrics['R2']
            }
            json.dump(metrics, f, indent=4)

    def save_model(self):
        """
        Save the model.
        """
        logging.info(f"Saving the model")
        os.makedirs(self.model_path, exist_ok=True)
        model_file_path = os.path.join(self.model_path, "model.joblib")
        joblib.dump(self.model, model_file_path)

@steps
def train_and_evaluate_model(config_path: str):
    """
    Create, train and evaluate the model.
    Args:
        config_path: path to the config file
    """
    try:
        model = Model(config_path)
        model.train()
        model.test()
        model.evaluate()
        model.save_parameters()
        model.save_metrics()
        model.save_model()
    except Exception as e:
        logging.error(f"Error while training and evaluating the model: {e}")
        raise e


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    default_config_path = os.path.join("config", "params.yaml")
    args.add_argument('--config', type=str, default=default_config_path)
    parsed_args = args.parse_args()
    train_and_evaluate_model(config_path=parsed_args.config)
