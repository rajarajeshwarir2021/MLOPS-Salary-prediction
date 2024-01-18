from pipelines.inference_pipeline import inference_pipeline

DUMMY_DATA = {
    "Gender": "Male",
    "Education_Level": "Bachelor's",
    "Job_Title": "Software Engineer",
    "Years_of_Experience": 15.0
}

if __name__ == '__main__':
    # Run the inference pipeline
    inference_pipeline(config_path="config/params.yaml", user_data=DUMMY_DATA)