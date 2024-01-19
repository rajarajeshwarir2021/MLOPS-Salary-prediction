import pytest

from pipelines.inference_pipeline import inference_pipeline
from src.get_metadata import GetInferenceData
from src.validate_input import NotInRange

input_data = {
    "incorrect_range":
    {"Gender": "Others",
    "Education_Level": "Doctorate",
    "Job_Title": "Teacher",
    "Years_of_Experience": 50
    },

    "correct_range":
    {"Gender": "Male",
    "Education_Level": "Bachelor's",
    "Job_Title": "Software Engineer",
    "Years_of_Experience": 10
    },

    "incorrect_col":
    {"Gendar": "Male",
    "Education Level": "Bachelor's",
    "Job_Tilte": "Software Engineer",
    "Years_of Experience": 10
    }
}

TARGET_VALUES = {
    "min": 350.0,
    "max": 250000.0
}

def test_form_response_correct_range(data=input_data["correct_range"]):
    inference_pipeline(config_path="config/params.yaml", user_data=data)
    res = GetInferenceData().get_inference_data()
    assert TARGET_VALUES["min"] <= res <= TARGET_VALUES["max"]

def test_form_response_incorrect_range(data=input_data["incorrect_range"]):
    with pytest.raises(NotInRange):
        inference_pipeline(config_path="config/params.yaml", user_data=data)
        res = GetInferenceData().get_inference_data()
