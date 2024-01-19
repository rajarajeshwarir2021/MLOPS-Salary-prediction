import argparse
import logging
import os
import pandas as pd
from zenml.client import Client

class GetInferenceData:
    """
    A class for fetching the inference result data from the Zenml.
    """
    def get_inference_data(self):
        """
        Fetching the result data from the Zenml.
        Returns: A float representing the inference result
        """
        logging.info(f"Fetching result data from Zenml")
        try:
            runs = Client().list_pipeline_runs()
            run_id = runs[0].artifact_versions.pop().id
            #print(run_id)
            artifact_metadata = Client().get_artifact_version(run_id)
            result = artifact_metadata.load()
            #print(result)
            return result
        except Exception as e:
            logging.error(f"Error while fetching result data from Zenml: {e}")
            raise e


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    default_config_path = os.path.join("config", "params.yaml")
    args.add_argument('--config', type=str, default=default_config_path)
    parsed_args = args.parse_args()
    GetInferenceData().get_inference_data()
