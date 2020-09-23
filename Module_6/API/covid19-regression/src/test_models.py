import joblib
import pandas as pd
import numpy as np
import os
from datetime import datetime

import utils

MODELS_FOLDER = os.path.join('..', 'models')
TIMESTAMP = datetime.strftime(datetime.now(), '%y-%m-%d')

# GPR - Navar
model_gpr = joblib.load(os.path.join(MODELS_FOLDER, 'model-{}-gpr-0.1.pkl'.format(TIMESTAMP)))

# Linear - Navar

model_linreg = joblib.load(os.path.join(MODELS_FOLDER, 'model-{}-linear-polynomial-0.1.pkl'.format(TIMESTAMP)))
# y_hat_ceara_linreg = model_linreg.predict(X_ceara) 

utils.forecast(model_linreg, future=10)