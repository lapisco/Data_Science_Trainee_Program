import joblib
from os import path
import numpy as np


DIR_NAME = path.dirname(__file__)
MODELS_FOLDER = path.join('.', 'models')
EXPERIMENT_NAME = path.join(MODELS_FOLDER, 'exp_01_default')

from decouple import config as cfg
TRANFORMER_NAME = cfg('TRANFORMER_NAME', cast=str)
MODEL_NAME = cfg('MODEL_NAME', cast=str)


def load_models():
    '''
    Load models routine
    '''
    tf = joblib.load(path.join(EXPERIMENT_NAME, TRANFORMER_NAME))
    model = joblib.load(path.join(EXPERIMENT_NAME, MODEL_NAME))
    
    return model, tf


def check_inputs(input):
    print(input)

    # check if is list
    if type(input) == list:
        if len(input) == 4:
            # turn strings into numbers
            input = [float(i) for i in input]
            return np.array(input).reshape(1,-1)
    
    else:
        return 205

    pass