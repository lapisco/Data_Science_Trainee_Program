import argparse
from flask import Flask, render_template, request, Response, send_file, jsonify
import joblib
import os
import sys
from datetime import datetime
sys.path.append('../templates')

import json
import pandas as pd

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import logging
filename_log = 'registry.log'
# if os.path.exists(os.path.join(os.path.dirname(__file__), filename_log)):
#     os.remove(os.path.join(os.path.dirname(__file__), filename_log))

logging.basicConfig(level=logging.INFO, filename=filename_log)
logging.info("Logging starting...")
logging.info("Server starting... at {}".format(datetime.now()))


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# BASE_DIR = '..'
# TEMPLATE_DIR = os.path.join(BASE_DIR, 'templates') 
TEST_DATA = 'test_data.json'

from decouple import config as cfg
MODELS_PATH = cfg('MODELS_PATH', cast=str)
PIPELINE_NAME = cfg('PIPELINE_NAME', cast=str)
DEBUG = cfg('DEBUG', default=False, cast=bool)
PORT = cfg('PORT', default=80, cast=int)
LOG = cfg('LOG', default=False, cast=bool)

app = Flask(__name__)

filepath = './.temp/'

# Load pipeline:
pipeline = joblib.load(os.path.join(MODELS_PATH, PIPELINE_NAME))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/churn-detector/pipeline', methods=['POST'])
def predict():  
    if request.method == 'POST':
        try:
            # Load json from the request object
            input_json = request.get_json()            
            # Parse json into a pandas DataFrame
            for user, features in input_json.items():
                df_features = pd.DataFrame([features])

            # Pipeline transformation
            X = pipeline.transform(df_features)

            return jsonify(user=user,
                            features=X.tolist(),
                            status=200,
                            message='Pipeline working')
        
        except:
            print("Unexpected error: {}".format(sys.exc_info()[0]))
            logging.info("Unexpected error: {}".format(sys.exc_info()[0]))

            return jsonify(message='error {}'.format(sys.exc_info()[0]), status=500)
        
        
@app.route('/churn-detector/pipeline/test', methods=['GET'])
def predict_test():  
    if request.method == 'GET':
        try:
            with open('output_data.json') as json_file:
                data = json.load(json_file)
            # Load json from the request object
            # json.load('output_data.json')
            # Parse json into a pandas DataFrame
            
            return jsonify(data)

            # return jsonify(user=user,
            #                 features=X.tolist(),
            #                 status=200,
            #                 message='Pipeline working')
        
        except:
            print("Unexpected error: {}".format(sys.exc_info()[0]))
            logging.info("Unexpected error: {}".format(sys.exc_info()[0]))

            return jsonify(message='error {}'.format(sys.exc_info()[0]), status=500)


def read_test_data():
    return pd.read_json(TEST_DATA, orient='records')

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--port', default=80, type=int)
    parser.add_argument('--host', default='0.0.0.0', type=str)
    args = vars(parser.parse_args()) 

    app.run(host=args['host'], port=args['port'], debug=DEBUG)