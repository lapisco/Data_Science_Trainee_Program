import joblib
import pandas as pd 
import numpy as np
import os 
from fbprophet import Prophet
from datetime import datetime
from statsmodels.tsa.api import Holt

import utils


# Global variables
DATA_FOLDER = os.path.join('..', 'data')
MODELS_FOLDER = os.path.join('..', 'models')
CEARA_DATA = 'ceara.csv'
TIMESTAMP = datetime.strftime(datetime.now(), '%y-%m-%d')

# Load dataset
from decouple import config as cfg
DATA_FOLDER_PROCESSED = cfg('DATA_FOLDER_PROCESSED', cast=str)

# Train ceara Model
ceara_df = pd.read_csv(os.path.join(DATA_FOLDER_PROCESSED, CEARA_DATA), index_col=0)
# ceara_df = pd.read_csv(os.path.join(DATA_FOLDER, 'ceara.csv'))

# Treinar so com os ultimos 9 dias
new_df = ceara_df.iloc[-9:].copy()
new_df.reset_index(inplace=True)
new_df.drop(labels=['cured', 'deaths', 'refuses', 'suspects'], axis=1, inplace=True)
new_df.rename(columns={'data': 'ds', 'cases': 'y'}, inplace=True)

##

print(new_df)

# # Preprocessing:
# ## Selector, index=df.index, columns=['sexo'])
# full_pipeline = joblib.load(os.path.join(MODELS_FOLDER, 'full_pipeline-0.1.pkl'))
# X = full_pipeline.transform(info_data_cleaned)
# y = user_data['status'].values

# print("X shape: {}".format(X.shape))
# print("y shape: {}".format(y.shape))

# Train Models
model = Prophet()
model.fit(new_df)

future = model.make_future_dataframe(periods=10)
print("Days ahead: {}".format(future.tail()))

forecast = model.predict(future)
print("Predict: {}".format(forecast))

# Workround to error TypeError: can't pickle _thread.RLock objects
# found at: https://github.com/facebook/prophet/issues/1361
model.stan_backend.logger = None
# Dumping the model
joblib.dump(model, os.path.join(MODELS_FOLDER, 'model-{}-prophet-0.1.pkl'.format(TIMESTAMP)))

# Train the Exponential from stats
new_df['y']=new_df['y']*0.05
# instatiate the mdel
new_df = utils.porra(ceara_df)

model_holt = Holt(new_df, exponential=True).fit(smoothing_level=0.8, smoothing_slope=0.2, optimized=False)
forecast_2 = model_holt.forecast(10).rename("Exponential trend")
print(forecast_2)

joblib.dump(model, os.path.join(MODELS_FOLDER, 'model-{}-holt-0.1.pkl'.format(TIMESTAMP)))





