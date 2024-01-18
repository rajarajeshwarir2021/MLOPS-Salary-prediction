from abc import ABC, abstractmethod
import json
import logging
import os


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

def read_json(file_path):
    """
    Read the json data.
    """
    with open(file_path, 'r') as f:
        schema = json.load(f)
    return schema

def get_schema(config_params):
    """
    Get the schema.
    """
    schema_path = config_params['preprocess_data_source']['dataset_schema_json']
    schema_file_path = os.path.join(schema_path, "dataset_schema.json")
    return read_json(schema_file_path)

def validate_input(dict_request, schema_path):
    """
    Validate the input.
    """
    def _validate_cols(col):
        schema = get_schema(schema_path)
        actual_cols = schema.keys()
        if col not in actual_cols:
            raise NotInFeatureColumn

    def _validate_values(col, val):
        schema = get_schema()
        if col in ["Gender", "Education_Level", "Job_Title"]:
            if not val in schema[col].values():
                raise NotInRange
        elif not (schema[col]["min"] <= float(dict_request[col]) <= schema[col]["max"]):
            raise NotInRange

    for col, val in dict_request.items():
        _validate_cols(col)
        _validate_values(col, val)

    return True


class ValidateIO(ABC):
    """
    Abstract class for validating the input or output data.
    """
    @abstractmethod
    def validate(self, data, config_params) -> bool:
        """
        Validate the data.
        Args:
            data: data to validate
            config_params: configuration parameters object
        Returns:
            A boolean
        """
        pass

class ValidateInput(ValidateIO):
    """
    A class for validating the input data.
    """
    def validate(self, data: dict, config_params: object) -> bool:
        """
        Validate the input data.
        Args:
            data: data to validate
            config_params: configuration parameters object
        Returns:
            A boolean
        """
        try:
            if validate_input(data, config_params):
                logging.info(f"Input validated")
                return True
        except Exception as e:
            logging.error(f"Error while validating input: {e}")
            raise e
