import argparse
import requests
from flask import Flask, render_template, request, Response, send_file, jsonify
from flask_jsonpify import jsonpify
from flask_cors import CORS
import joblib
import os
import sys
import datetime
sys.path.append('../templates')
sys.path.append('..')
import utils

import json
import pandas as pd
import numpy as np
from statsmodels.tsa.api import Holt

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import logging
filename_log = 'registry.log'
# if os.path.exists(os.path.join(os.path.dirname(__file__), filename_log)):
#     os.remove(os.path.join(os.path.dirname(__file__), filename_log))

logging.basicConfig(level=logging.INFO, filename=filename_log)
logging.info("Logging starting...")
logging.info("Server starting... at {}".format(datetime.datetime.now()))


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from decouple import config as cfg
MODELS_PATH = cfg('MODELS_PATH', cast=str)
MODELS_PATH = 'models/'

MODEL_NAME = cfg('MODEL_NAME', cast=str)
MODEL_HOLT_NAME = cfg('MODEL_EXPONENTIAL_NAME', cast=str)
MODEL_POLYNOMIAL_NAME = cfg('MODEL_POLYNOMIAL_NAME', cast=str)
MODEL_POLYNOMIAL_METADATA = cfg('MODEL_POLYNOMIAL_METADATA', cast=str)
MODEL_REGRESSION_NAME = cfg('MODEL_REGRESSION_NAME', cast=str)
MODEL_REGRESSION_METADATA = cfg('MODEL_REGRESSION_METADATA', cast=str)
MODEL_GPR_NAME = cfg('MODEL_GPR_NAME', cast=str)
MODEL_GPR_METADATA = cfg('MODEL_GPR_METADATA', cast=str)
DEBUG = cfg('DEBUG', default=False, cast=bool)
PORT_MODEL = cfg('PORT_MODEL', default=80, cast=int)
LOG = cfg('LOG', default=False, cast=bool)
# DATA_PATH = cfg('DATA_PATH', cast=str)
DATA_PATH = os.path.join('data', 'processed')

MODEL_OBITO_DATE= cfg('MODEL_OBITO_DATE', cast=str)

URL_SERVICE_GET_DATA_WORLD = cfg('URL_SERVICE_GET_DATA_WORLD', default=False, cast=str)
URL_SERVICE_GET_DATA_WORLD = URL_SERVICE_GET_DATA_WORLD + '/api/covid19stats/historyByCountry?Country='

URL_SERVICE_GET_DATA_CEARA = cfg('URL_SERVICE_GET_DATA_CEARA', default=False, cast=str)
URL_SERVICE_GET_DATA_CEARA = URL_SERVICE_GET_DATA_CEARA + '/api/covid19stats/historyByBrState?State=CE'

URL_SERVICE_GET_DATA_STATES = cfg('URL_SERVICE_GET_DATA_STATES', default=False, cast=str)
URL_SERVICE_GET_DATA_STATES = URL_SERVICE_GET_DATA_STATES + '/api/covid19stats/historyByBrState?State='

DAYS_TO_TRAIN = cfg('DAYS_TO_TRAIN', default=10, cast=int)


# PIPELINE_ENDPOINT = cfg('PIPELINE_ENDPOINT', cast=str)
# print(PIPELINE_ENDPOINT)

# LOCAL VARIABLES
VARIABLES_FBPROPHET = ['ds', 'yhat', 'yhat_lower', 'yhat_upper']

CEARA_DATA = 'ceara.csv'


app = Flask(__name__)
CORS(app, support_credentials=True)

# CREATE TEMP FOLDER
TEMPFOLDER = './.temp/'
if not os.path.exists(TEMPFOLDER):
    os.mkdir(TEMPFOLDER)

# Load model:
model = joblib.load(os.path.join(MODELS_PATH, MODEL_NAME))

# models = {
#     'linear-polynomial': joblib.load(os.path.join(MODELS_PATH, MODEL_POLYNOMIAL_NAME)),
#     'gpr': joblib.load(os.path.join(MODELS_PATH, MODEL_GPR_NAME)),
#     'linear-regression': joblib.load(os.path.join(MODELS_PATH, MODEL_REGRESSION_NAME)),
# }

# models_metadata = {
#     'linear-polynomial': joblib.load(os.path.join(MODELS_PATH, MODEL_POLYNOMIAL_METADATA)),
#     'gpr': joblib.load(os.path.join(MODELS_PATH, MODEL_GPR_METADATA)),
#     'linear-regression': joblib.load(os.path.join(MODELS_PATH, MODEL_REGRESSION_METADATA)),
# }

# models_details = {
#     'linear-polynomial': "Liner Regression + Polynomial Features degree 4",
#     'gpr': "Gaussian Process Regressor - Kernel RBF",
#     'linear-regression': "Liner Regression - degree 1"
# }

# models_names = {
#     'linear-polynomial': MODEL_POLYNOMIAL_NAME,
#     'gpr': MODEL_GPR_NAME,
#     'linear-regression': MODEL_REGRESSION_NAME
# }

# model_holt = joblib.load(os.path.join(MODELS_PATH, MODEL_HOLT_NAME))
ceara_df = pd.read_csv(os.path.join(DATA_PATH, 'CE.csv'))
# ceara_df_nauber = pd.read_csv(os.path.join('data', 'raw', 'ceara_Nauber.csv'))

models, models_metadata, models_details, models_names = utils.load_models()
models_obito, models_obito_metadata, models_obito_details, models_obito_names = utils.load_models(basename='model-obito-', model_date=MODEL_OBITO_DATE)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/covid19-regression-linear/predict', methods=['POST'])
def predict():  
    if request.method == 'POST':
        print("Request recieved")
        try:            
            # PASS TO THE PIPELINE CONTAINER
            # output_pipeline = requests.post(url=PIPELINE_ENDPOINT,json=input_json)

            # Predict with the model
            var_future = int(request.args['future'])
            print("Future: {}".format(var_future))
            future = model.make_future_dataframe(periods=var_future)
            forecast = model.predict(future)
            print(forecast[VARIABLES_FBPROPHET].tail()) 
            output = {'ds': forecast['ds'].tolist(),
                     'yhat': forecast['yhat'].tolist(),
                     'yhat_lower': forecast['yhat_lower'].tolist(),
                     'yhat_upper': forecast['yhat_upper'].tolist()}
                    
            return jsonify( predict=output,
                            model_name=MODEL_NAME,
                            method='linear',
                            status=200,
                            message='Model working')

        except:
            print("Unexpected error: {}".format(sys.exc_info()[0]))
            logging.info("Unexpected error: {}".format(sys.exc_info()[0]))

            return jsonify(message='error {}'.format(sys.exc_info()[0]), status=500)

        
@app.route('/covid19-regression-linear/predict/test', methods=['GET'])
def predict_test():  
    if request.method == 'GET':
        print("testing...")
        try:
            with open('output_data.json') as json_file:
                data = json.load(json_file)
            
            return jsonify(data)
        
        except:
            print("Unexpected error: {}".format(sys.exc_info()[0]))
            logging.info("Unexpected error: {}".format(sys.exc_info()[0]))

            return jsonify(message='error {}'.format(sys.exc_info()[0]), status=500)



@app.route('/covid19-regression-exponential/predict', methods=['POST'])
def predict_exponential():  
    if request.method == 'POST':
        print("Request recieved")
        try:            
            # PASS TO THE PIPELINE CONTAINER
            # output_pipeline = requests.post(url=PIPELINE_ENDPOINT,json=input_json)

            # Predict with the model
            var_future = int(request.args['future'])
            var_state = request.args['state']
            if not 'label' in request.args:
                var_label = 'cases'
                base = datetime.datetime.today()
                # future = [(base - datetime.timedelta(days=x)).replace(tzinfo=datetime.timezone.utc).timestamp() for x in range(var_future)]
                future = utils.dates_to_future(days=var_future)
                future = utils.format_date(future)
                # TODO: there should not be traning during predition.
                # this is done while we found and workaround to stats models
                # Load data
                df_state = utils.download_state(state=var_state)
                porra_df = utils.porra(df_state.iloc[-DAYS_TO_TRAIN:], var_label)
            else:
                var_label = request.args['label']
                base = datetime.datetime.today()
                # future = [(base - datetime.timedelta(days=x)).replace(tzinfo=datetime.timezone.utc).timestamp() for x in range(var_future)]
                future = utils.dates_to_future(days=var_future)
                future = utils.format_date(future)
                # TODO: there should not be traning during predition.
                # this is done while we found and workaround to stats models
                # Load data
                df_state = utils.download_state(state=var_state)
                if var_label == 'deaths':
                    porra_df = utils.porra(df_state, var_label)
                else:
                    porra_df = utils.porra(df_state.iloc[-DAYS_TO_TRAIN:], var_label)

            print("Future: {}".format(var_future))
            
            model_holt = Holt(porra_df['y'], exponential=True).fit(smoothing_level=0.5, smoothing_slope=0.05, optimized=False)
            forecast = model_holt.forecast(var_future)
            
            # Check if predictions are belows yesterday
            last_value = porra_df.iloc[-1]['y']
            forecast = utils.rescale_yhat(forecast.values, last_value)

            output = {'ds': future,
                     'yhat': forecast.tolist()}
                     
                        
            return jsonify( predict=output,
                            last_value=last_value.tolist(),
                            status=200,
                            message='Model working',
                            model_name=MODEL_HOLT_NAME,
                            method='Holt')

        except:
            print("Unexpected error: {}".format(sys.exc_info()[0]))
            logging.info("Unexpected error: {}".format(sys.exc_info()[0]))

            return jsonify(message='error {}'.format(sys.exc_info()[0]), status=500)


@app.route('/covid19-regression-exponential/predict/test', methods=['GET'])
def predict_exponential_test():  
    if request.method == 'GET':
        print("testing...")
        try:
            with open('output_data_exponential.json') as json_file:
                data = json.load(json_file)
            
            return jsonify(data)
        
        except:
            print("Unexpected error: {}".format(sys.exc_info()[0]))
            logging.info("Unexpected error: {}".format(sys.exc_info()[0]))

            return jsonify(message='error {}'.format(sys.exc_info()[0]), status=500)



@app.route('/covid19-regression/predict', methods=['POST'])
def predict_models():  
    if request.method == 'POST':
        print("Request recieved")
        try:            
            # PASS TO THE PIPELINE CONTAINER
            # output_pipeline = requests.post(url=PIPELINE_ENDPOINT,json=input_json)

            # Predict with the model
            var_state = request.args['state']
            var_future = int(request.args['future'])
            var_model_method = request.args['method']
            
            print("Future: {}".format(var_future))
            
            # base = datetime.datetime.today()
            # future = [(base - datetime.timedelta(days=x)).replace(tzinfo=datetime.timezone.utc).timestamp() for x in range(var_future)]
            # this is done while we found and workaround to stats models
            dayone = models_metadata[var_state][var_model_method]['dayone']
            predictions = utils.forecast(models[var_state][var_model_method], future=var_future, dayone=dayone, date_string='%d/%m/%Y')
            
            # Check if predictions are belows yesterday
            df_state = utils.download_state(state=var_state)
            last_value = df_state['cases'].iloc[-1]
            models_metadata[var_state][var_model_method]['last_value'] = last_value.tolist()
            # last_value = models_metadata[var_state][var_model_method]['last_value']
            predictions['yhat'] = utils.rescale_yhat(predictions['yhat'], last_value)

            # Put it into a dictionary
            output = {'ds': predictions['ds'].tolist(),
                     'yhat': predictions['yhat'].tolist()}
                        

            return jsonify( predict=output,
                            metadata=models_metadata[var_state][var_model_method],
                            status=200,
                            message='Model working',
                            model_name=models_names[var_state][var_model_method],
                            model_detais=models_details[var_state][var_model_method],
                            method=var_model_method)

        except:
            print("Unexpected error: {}".format(sys.exc_info()[0]))
            logging.info("Unexpected error: {}".format(sys.exc_info()[0]))


            return jsonify(message='error {}'.format(sys.exc_info()[0]), status=500)


@app.route('/covid19-regression/predict-death', methods=['POST'])
def predict_death_models():  
    if request.method == 'POST':
        print("Request recieved")
        try:            
            # PASS TO THE PIPELINE CONTAINER
            # output_pipeline = requests.post(url=PIPELINE_ENDPOINT,json=input_json)

            # Predict with the model
            var_state = request.args['state']
            var_future = int(request.args['future'])
            var_model_method = request.args['method']
            
            print("Future: {}".format(var_future))
            
            # base = datetime.datetime.today()
            # future = [(base - datetime.timedelta(days=x)).replace(tzinfo=datetime.timezone.utc).timestamp() for x in range(var_future)]
            # this is done while we found and workaround to stats models
            dayone = models_obito_metadata[var_state][var_model_method]['dayone']
            predictions = utils.forecast(models_obito[var_state][var_model_method], future=var_future, dayone=dayone, date_string='%d/%m/%Y')
            
            # Check if predictions are belows yesterday
            df_state = utils.download_state(state=var_state)
            last_value = df_state['deaths'].iloc[-1]
            models_obito_metadata[var_state][var_model_method]['last_value'] = last_value.tolist()
            # last_value = models_obito_metadata[var_state][var_model_method]['last_value']
            predictions['yhat'] = utils.rescale_yhat(predictions['yhat'], last_value)

            # Put it into a dictionary
            output = {'ds': predictions['ds'].tolist(),
                     'yhat': predictions['yhat'].tolist()}
                        

            return jsonify( predict=output,
                            metadata=models_obito_metadata[var_state][var_model_method],
                            status=200,
                            message='Model working',
                            model_name=models_obito_names[var_state][var_model_method],
                            model_detais=models_obito_details[var_state][var_model_method],
                            method=var_model_method)

        except:
            print("Unexpected error: {}".format(sys.exc_info()[0]))
            logging.info("Unexpected error: {}".format(sys.exc_info()[0]))


            return jsonify(message='error {}'.format(sys.exc_info()[0]), status=500)



@app.route('/covid19-regression/data-process', methods=['POST'])
def data_process():  
    if request.method == 'POST':
        print("Request recieved")
        try:     
            # TODO: TREAT inputs. all to lower case or upper case. Standard       
            var_country= str(request.args['country'])

            if var_country in utils.STATES:
                df_state = utils.download_state(URL=URL_SERVICE_GET_DATA_STATES, state=var_country)
                X = df_state['cases'].values.tolist()
                increments, smoothed_increments, cummulative = utils.get_increments(X)

                date_list = df_state[df_state['cases'] != 0].index.to_list()
                ds = utils.format_date(date_list, date_string_input='%d/%m/%Y')
            else:
                r = requests.get(URL_SERVICE_GET_DATA_WORLD + var_country)
                # Create Dataframe and populate it
                dates = []
                for day in r.json():
                    dates.append(day['data'])
                    
                confirmed_df = pd.DataFrame(columns=dates, index=[var_country])
                cases = []
                deaths = []
                recovered = []

                for day in r.json():
                    cases.append(day['cases'])
                    actual_day = day['data']
                    confirmed_df.loc[var_country, day['data']] = day['cases']
                print('Finished this country: {} \n ================='.format(var_country))
                print(confirmed_df)
            
                # Calculate stuff:
                X = confirmed_df.loc[var_country].values.tolist()
                increments, smoothed_increments, cummulative = utils.get_increments(X)

                # Generate date output
                date_list = confirmed_df.columns
                ds = utils.format_date(confirmed_df.columns.to_list(), date_string_input='%m/%d/%y')
            
            output = {'ds': ds,
                    'cummulative': np.log10(cummulative).tolist(),
                    'increments': increments,
                    'smoothed_increments': np.log10(smoothed_increments).tolist()}
                        
            return jsonify( data_process=output,
                            status=200,
                            message='log conversion working',
                            details='''In output we have: 
                                - cummulative in log10; 
                                - increments in actual value; 
                                - smoothed_increments in log10. 
                                Moving average with window set to 7 in increments data
                                '''
                            )

        except:
            print("Unexpected error: {}".format(sys.exc_info()[0]))
            logging.info("Unexpected error: {}".format(sys.exc_info()[0]))

            return jsonify(message='error {}'.format(sys.exc_info()[0]), status=500)


@app.route('/covid19-regression/exponential-graph', methods=['POST'])
def exponential_graph():  
    if request.method == 'POST':
        print("Request recieved")
        try:     
            # TODO: TREAT inputs. all to lower case or upper case. Standard       
            var_state= str(request.args['state'])

            if var_state in utils.STATES:
                exp_01 = joblib.load(os.path.join(MODELS_PATH, var_state, 'exp-first-7'))
                exp_02 = joblib.load(os.path.join(MODELS_PATH, var_state, 'exp-first-14'))

                output = {
                    'exponential_01': utils.create_output_exponential_graphs(exp_01),
                    'exponential_02': utils.create_output_exponential_graphs(exp_02)
                }
                            
                return jsonify( data_process=output,
                                status=200,
                                )

            else:
                return jsonify( status=500,
                                message='The state or country not found',
                                )
                
        except:
            print("Unexpected error: {}".format(sys.exc_info()[0]))
            logging.info("Unexpected error: {}".format(sys.exc_info()[0]))

            return jsonify(message='error {}'.format(sys.exc_info()[0]), status=500)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--port', default=5000, type=int)
    parser.add_argument('--host', default='0.0.0.0', type=str)
    args = vars(parser.parse_args()) 

    app.run(host=args['host'], port=args['port'], debug=DEBUG)
