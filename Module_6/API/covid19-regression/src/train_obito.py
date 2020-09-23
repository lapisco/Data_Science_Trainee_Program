import joblib
import pandas as pd 
import numpy as np
import os 
from datetime import datetime

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.linear_model import LinearRegression, BayesianRidge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline

import utils
import requests, json

# Global variables
MODELS_FOLDER = os.path.join('..', 'models')
CEARA_DATA = 'ceara.csv'
TIMESTAMP = datetime.strftime(datetime.now(), '%y-%m-%d')

from decouple import config as cfg
DATA_FOLDER_PROCESSED = cfg('DATA_FOLDER_PROCESSED', cast=str)
DAYS_TO_TRAIN = cfg('DAYS_TO_TRAIN', default=10, cast=int)

# CREATE TEMP FOLDER
TEMPFOLDER = './.temp/'
if not os.path.exists(TEMPFOLDER):
    os.mkdir(TEMPFOLDER)

# Loda data from state
url = 'http://lapisco.fortaleza.ifce.edu.br:3011/api/covid19stats/listBrStates'
r = requests.get(url)
states = {}

for state in r.json():
    print('Training model for state: {}'.format(state['uf']))
    df_state = utils.download_state(state=state['uf'])
    # df_state = df_state.iloc[-DAYS_TO_TRAIN:]

    print(df_state.tail())

    # print(df_state)
    df_filtered = df_state[df_state['cases'] != 0]
    dayone = df_state[df_state['cases'] != 0].index[0]
    days = np.array(utils.count_days(dayone=dayone, date_string='%d/%m/%Y'))
    X = days.reshape(-1,1)
    # y = df_state['cases'].values.reshape(-1,1)
    y = df_filtered['deaths'].values.reshape(-1,1)

    print('{} - dayone: {}'.format(state, dayone))
    # print('Print inputs/outputs: \n X: {} \n y: {}' .format(X, y))
    print('Print inputs/outputs shapes: \n X: {} \n y: {}' .format(X.shape, y.shape))
    X, y = utils.check_inputs(X, y)
    print('Print inputs/outputs shapes corrected: \n X: {} \n y: {}' .format(X.shape, y.shape))


    model_gpr = GaussianProcessRegressor()
    model_gpr.fit(X, y)

    poly = PolynomialFeatures(degree=4)
    model_linreg = LinearRegression()
    model_full_linreg = Pipeline([
                    ('polynomial-features', poly),
                    ('regressor', model_linreg)
                    ])

    model_full_linreg.fit(X, y)

    model_linreg_simple = LinearRegression()
    model_linreg_simple.fit(X, y)

    # Predicting
    y_hat_gpr = model_gpr.predict(X)

    y_hat_linreg = model_full_linreg.predict(X)

    y_hat_linreg_simple = model_linreg_simple.predict(X)

    mse = {'GPR': mean_squared_error(y, y_hat_gpr),
            'Linear-Polynomial': mean_squared_error(y, y_hat_linreg),
            'Linear-Regression': mean_squared_error(y, y_hat_linreg_simple),
    }

    error_str = ''' Error rate: \n
                GPR: {} \n
                Linear-Polynomial: {} \n
                Linear-Regression: {} \n
                '''.format(mse['GPR'], mse['Linear-Polynomial'], mse['Linear-Regression'])

    model_metadata_gpr = {
        'mse': mse['GPR'],
        'fit_date': TIMESTAMP,
        'dayone': dayone,
        'last_value': y[-1].tolist()
    }

    model_metadata_linreg = {
        'mse': mse['Linear-Polynomial'],
        'fit_date': TIMESTAMP,
        'dayone': dayone,
        'last_value': y[-1].tolist()
    }

    model_metadata_linreg_simple = {
        'mse': mse['Linear-Regression'],
        'fit_date': TIMESTAMP,
        'dayone': dayone,
        'last_value': y[-1].tolist()
    }


    print(error_str)

    # Check if State`s folder exists
    if not os.path.exists(os.path.join(MODELS_FOLDER, state['uf'])):
        os.mkdir(os.path.join(MODELS_FOLDER, state['uf']))

    # Dumping models
    joblib.dump(model_gpr, os.path.join(MODELS_FOLDER, state['uf'], 'model-obito-{}-{}-gpr-0.1.pkl'.format(state['uf'], TIMESTAMP)))
    joblib.dump(model_metadata_gpr, os.path.join(MODELS_FOLDER, state['uf'], 'model-obito-{}-{}-gpr-metadata-0.1.pkl'.format(state['uf'], TIMESTAMP)))

    joblib.dump(model_full_linreg, os.path.join(MODELS_FOLDER, state['uf'], 'model-obito-{}-{}-linear-polynomial-0.1.pkl'.format(state['uf'], TIMESTAMP)))
    joblib.dump(model_metadata_linreg, os.path.join(MODELS_FOLDER, state['uf'], 'model-obito-{}-{}-linear-polynomial-metadata-0.1.pkl'.format(state['uf'], TIMESTAMP)))

    joblib.dump(model_linreg_simple, os.path.join(MODELS_FOLDER, state['uf'], 'model-obito-{}-{}-linear-regression-0.1.pkl'.format(state['uf'], TIMESTAMP)))
    joblib.dump(model_metadata_linreg_simple, os.path.join(MODELS_FOLDER, state['uf'], 'model-obito-{}-{}-linear-regression-metadata-0.1.pkl'.format(state['uf'], TIMESTAMP)))






