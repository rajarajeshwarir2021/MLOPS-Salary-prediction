import os
from flask import Flask, render_template, request, jsonify

from pipelines.inference_pipeline import inference_pipeline
from src.get_metadata import GetInferenceData
from src.read_config import ReadConfig

DUMMY_DATA = {
    "Gender": "Male",
    "Education_Level": "Bachelor's",
    "Job_Title": "Software Engineer",
    "Years_of_Experience": 15.0
}

WEB_APP_DATA = {
    'Gender': 'Male',
    'Education': "Bachelor's",
    'Job_Title': 'Software Engineer',
    'Years_of_Experience': '10'
}

config = ReadConfig(config_path="config/params.yaml")
params = config.read_params()

static_dir = params['web_app']['static_dir']
templates_dir = params['web_app']['template_dir']

app = Flask(__name__, static_folder=static_dir, template_folder=templates_dir)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            if request.form:
                data_req = dict(request.form)
                # Run the inference pipeline
                inference_pipeline(config_path="config/params.yaml", user_data=data_req)
                response = GetInferenceData().get_inference_data()
                print(response)
                return render_template('index.html', response=response)
        except Exception as e:
            print(e)
            error_message = {"error": e}
            return render_template('error.html', error=error_message)
    else:
        return render_template('index.html')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)