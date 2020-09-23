import joblib
import pandas as pd 
import numpy as np
import os 
from fbprophet import Prophet
from datetime import datetime
from sklearn.metrics import mean_squared_error
import utils


# Global variables
DATA_FOLDER = os.path.join('..', 'data')
MODELS_FOLDER = os.path.join('..', 'models')
CEARA_DATA = 'ceara.csv'
TIMESTAMP = datetime.strftime(datetime.now(), '%y-%m-%d')

# Load dataset
ceara_df = pd.read_csv(os.path.join(DATA_FOLDER, CEARA_DATA))
print(ceara_df)

# Loading the model
model = joblib.load(os.path.join(MODELS_FOLDER, 'model-{}-prophet-0.1.pkl'.format(TIMESTAMP)))

future = model.make_future_dataframe(periods=10)
print("Days ahead: {}".format(future.tail()))

forecast = model.predict(future)
print("Predict: {}".format(forecast))

# Exponential

model_holt = joblib.load(os.path.join(MODELS_FOLDER, 'model-{}-holt-0.1.pkl'.format(TIMESTAMP)))

forecast = model_holt.forecast(steps=10)
print("Predict: {}".format(forecast))

# ########Prepare data form me
days = utils.count_days(ceara_df, date_string='%Y-%m-%d %H:%M')
X_ceara = days[:-4].reshape(-1,1)
y_ceara = ceara_df.values.reshape(-1,1)

# GPR - Navar

model_gpr = joblib.load(os.path.join(MODELS_FOLDER, 'model-{}-gpr-0.1.pkl'.format(TIMESTAMP)))
y_hat_ceara_gpr = model_gpr.predict(X_ceara)

# Linear - Navar

model_linreg = joblib.load(os.path.join(MODELS_FOLDER, 'model-{}-linear-polynomial-0.1.pkl'.format(TIMESTAMP)))
y_hat_ceara_linreg = model_linreg.predict(X_ceara)


#

mse = {'GPR': mean_squared_error(y_ceara, y_hat_ceara_gpr),
        'Linear-Polynomial': mean_squared_error(y_ceara, y_hat_ceara_linreg),
}

error_str = ''' Error rate: \n
            GPR: {} \n
            Linear-Polynomial: {}
            '''.format(mse['GPR'], mse['Linear-Polynomial'])

print(error_str)






