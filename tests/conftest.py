import json
import yaml
import pytest

@pytest.fixture(scope="session")
def config(config_path="config/params.yaml"):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

@pytest.fixture(scope="session")
def schema_in(schema_path="schema/dataset_schema.json"):
    with open(schema_path, 'r') as f:
        schema = json.load(f)
    return schema