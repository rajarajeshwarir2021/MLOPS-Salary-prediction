import argparse
import json
import logging
import os
from typing import Dict

from src.read_config import ReadConfig


class NotInRange(Exception):
    """
    Raised when values entered are not in range.
    """
    def __init__(self, message="Values entered are not in range"):
        self.message = message
        super().__init__(self.message)

class NotInFeatureColumn(Exception):
    """
    Raised when values entered are not in feature columns.
    """
    def __init__(self, message="Values entered are not in feature columns"):
        self.message = message
        super().__init__(self.message)


class ValidateUserInput:
    """
    A class for validating the input data.
    """
    def __init__(self, data: dict, config_params: object):
        """
        Validate the input data.
        Args:
            data: data to validate
            config_params: configuration parameters object
        Returns:
            A boolean
        """
        self.data = data
        self.config_params = config_params

    def validate(self) -> bool:
        try:
            if self.validate_input(self.data, self.config_params):
                logging.info(f"Input validated")
                return True
        except Exception as e:
            logging.error(f"Error while validating input: {e}")
            raise e

    @staticmethod
    def validate_input(dict_data, config_params):
        """
        Validate the input.
        """

        def _validate_cols(col):
            schema = ValidateUserInput.get_schema(config_params)
            actual_cols = schema.keys()
            if col not in actual_cols:
                raise NotInFeatureColumn

        def _validate_values(col, val):
            schema = ValidateUserInput.get_schema(config_params)
            if col in ["Gender", "Education_Level", "Job_Title"]:
                if not val in schema[col].values():
                    raise NotInRange
            elif not (schema[col]["min"] <= float(dict_data[col]) <= schema[col]["max"]):
                raise NotInRange

        for col, val in dict_data.items():
            _validate_cols(col)
            _validate_values(col, val)

        return True

    @staticmethod
    def get_schema(config_params):
        """
        Get the schema.
        """
        schema_path = config_params['preprocess_data_source']['dataset_schema_json']
        schema_file_path = os.path.join(schema_path, "dataset_schema.json")
        return ValidateUserInput.read_json(schema_file_path)

    @staticmethod
    def read_json(file_path):
        """
        Read the json data.
        """
        with open(file_path, 'r') as f:
            schema = json.load(f)
        return schema

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    default_config_path = os.path.join("config", "params.yaml")
    args.add_argument('--config', type=str, default=default_config_path)
    parsed_args = args.parse_args()
    DUMMY_DATA = {
        "Gender": "Male",
        "Education_Level": "Bachelor's",
        "Job_Title": "Software Engineer",
        "Years_of_Experience": 10
    }
    config = ReadConfig(config_path=parsed_args.config)
    params = config.read_params()
    val_in = ValidateUserInput(DUMMY_DATA, params)
    result = val_in.validate()
    print(result)
