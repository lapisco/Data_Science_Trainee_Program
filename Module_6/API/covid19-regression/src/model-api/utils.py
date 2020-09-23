from datetime import datetime, timedelta
from os import path
import numpy as np
import pandas as pd
import requests
import json, joblib
import collections

DATE_LIST = [
      "03-25-20",
      "03-26-20",
      "03-27-20",
]

LABELS_TO_DROP_OLD = ['cured', 'deaths', 'refuses', 'suspects']
LABELS_TO_DROP = ['internadosDomiciliar', 'internadosEnfermaria',
       'internadosUTI', 'percentual_cured', 'percentual_deaths',
       'percentual_internados', 'percentual_internadosDomiciliar',
       'percentual_internadosEnfermaria', 'percentual_internadosUTI',
       'total_internados_DB']

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

MODEL_DATE= cfg('MODEL_DATE', cast=str)
MODEL_OBITO_DATE= cfg('MODEL_OBITO_DATE', cast=str)
VERSION= cfg('VERSION', cast=str)


STATES = ['AC', 'AL', 'AM', 'AP', 'BA', 'CE',
         'DF', 'ES', 'GO', 'MA', 'MT', 'MS',
         'MG', 'PA', 'PB', 'PR', 'PE', 'PI',
         'RJ', 'RN', 'RO', 'RS', 'RR', 'SC',
         'SE', 'SP', 'TO']

MODELS = ['linear-polynomial', 'gpr', 'linear-regression']

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

MODELS_DETAILS = {
    'linear-polynomial': "Liner Regression + Polynomial Features degree 4",
    'gpr': "Gaussian Process Regressor - Kernel RBF",
    'linear-regression': "Liner Regression - degree 1"
}

# models_names = {
#     'linear-polynomial': MODEL_POLYNOMIAL_NAME,
#     'gpr': MODEL_GPR_NAME,
#     'linear-regression': MODEL_REGRESSION_NAME
# }



def get_labels(df):
    df_out = df.loc[df != 0]

    return df_out.values

def count_days(dayone='03-14-2020', date_string='%m-%d-%y'):
    dayone = datetime.strptime(dayone, date_string)
    days_list = np.arange(1, (datetime.today() - dayone).days + 2, 1).tolist()

    return days_list

def days_until_now(dayone='03-14-2020', date_string='%m/%d/%y'):
    dayone = datetime.strptime(dayone, date_string)
    days_list = np.arange(1, (datetime.today() - dayone).days + 2, 1).tolist()

    return len(days_list)


def dates_to_future(days=1):
    base = datetime.today()
    date_list = [datetime.strftime(base + timedelta(days=x), '%m-%d-%y') for x in range(days)]

    return date_list


def forecast(model, future=1, dayone='03-14-2020', date_string='%m-%d-%Y'):
    # Count the number of days up now
    days_before = days_until_now(dayone=dayone, date_string=date_string)
    # Count the days ahead from now
    days_future = [x for x in range(days_before, future + days_before + 2)]
    base = datetime.today()
    date_list = [datetime.strftime(base + timedelta(days=x), '%m-%d-%y') for x in range(future + 1)]

    # map_iterator = map(lambda x: datetime.strptime(x, date_string).day, days)
    # days_array = np.array(list(map_iterator))

    # predict with the model
    y_hat = model.predict(np.array(days_future).reshape(-1,1))

    out = pd.DataFrame(columns=['ds', 'yhat'])
    out['ds'] = format_date(date_list)
    out['yhat'] = np.delete(y_hat, 0)

    return out


# def porra(df):
#     new_df = df.iloc[-58:].copy()
#     # new_df = df.copy()
#     new_df.reset_index(inplace=True)
#     new_df.drop(labels=['cured', 'deaths', 'refuses', 'suspects'], axis=1, inplace=True)
#     new_df.rename(columns={'data': 'ds', 'cases': 'y'}, inplace=True)

#     new_df['y']=new_df['y']*0.05

#     return new_df

def porra(df, label='cases'):
    df_new = df.loc[df['cases'] != 0]
    # df_new.drop(labels=['cured', 'deaths', 'refuses', 'suspects'], axis=1, inplace=True)
    # df_new.drop(labels=LABELS_TO_DROP + LABELS_TO_DROP_OLD, axis=1, inplace=True)
    df_new = df_new[label].to_frame()
    # df_new = df_new['cases'].reset_index().drop(labels=['index'], axis=1)
    df_new.rename(columns={label: 'y'}, inplace=True)

    # check if there are zeros
    df_new[df_new['y'] == 0] = 1

    return df_new


def format_date(date_list, date_string_input='%m-%d-%y'):
    dates = [datetime.strptime(x, date_string_input) for x in date_list]
    dates_tormated = [x.strftime('%a, %d %b %Y %H:%m:%S') for x in dates]

    return dates_tormated


def get_increments(time_series, ceara=False):
    increments = []
    cummulative = []
    state = 0
    if not ceara:
        pass
#         time_series = np.sum(np.array(time_series), axis=0)
    else:
        pass
#     print(time_series)
    for i in time_series:
        if i==0:
            continue
        else:
            increments.append(i-state)
            cummulative.append(i)
            state = i
    window_size = 7
    smoothed_increments = []
    for i in range(len(increments)):
        smoothed_increments.append(np.abs(np.mean(increments[np.max([i-window_size,0]):i+1])+1))

    

    return increments, smoothed_increments, cummulative

def download_ceara(url):
    r = requests.get(url)
    d = json.loads(r.text)
    with open('ceara.json', 'w', encoding='utf-8') as f:
        json.dump(d, f, ensure_ascii=False, indent=4)
    
    df = pd.read_json('ceara.json')
    df.set_index(df['data'], inplace=True)
    df.drop(columns=['data'], inplace=True)
    
    return df


def download_state(URL='http://lapisco.fortaleza.ifce.edu.br:3022/api/covid19stats/historyByBrState?State=', state='CE', dump_folder='./.temp', save=False):
    URL = URL + state
    r = requests.get(URL)
    d = json.loads(r.text)
    with open(path.join(dump_folder, '{}.json'.format(state)), 'w', encoding='utf-8') as f:
        json.dump(d, f, ensure_ascii=False, indent=4)
    
    df = pd.read_json(path.join(dump_folder, '{}.json'.format(state)))
    df.set_index(df['data'], inplace=True)
    df.drop(columns=['data'], inplace=True)
#     print(df)
    if save:
        df.to_csv(path.join(dump_folder, '{}.csv'.format(state)))
    
    return df


def load_models(basename='model-', model_date=MODEL_DATE):
    models = collections.defaultdict(dict)
    models_metadata = collections.defaultdict(dict)
    models_details = collections.defaultdict(dict)
    models_names = collections.defaultdict(dict)

    for state in STATES:
        for model in MODELS:
            print('Loading model {} from state {}'.format(model, state))
            # models
            model_name = '{}{}-{}-{}-{}.pkl'.format(basename, state, model_date, model, VERSION)
            models[state][model] = joblib.load(path.join(MODELS_PATH, state, model_name))
            models_names[state][model] = model_name
            # metadata
            models_metadata_name = '{}{}-{}-{}-metadata-{}.pkl'.format(basename, state, model_date, model, VERSION)
            models_metadata[state][model] = joblib.load(path.join(MODELS_PATH, state, models_metadata_name))
            # details
            models_details[state][model] = MODELS_DETAILS[model]


    return models, models_metadata, models_details, models_names


def rescale_yhat(y_hat, last_value, factor=0.2):
    if y_hat[0] < last_value:
        diff = np.abs(y_hat[0] - last_value)
        y_hat = y_hat + diff*(1 + factor)

    return y_hat



def create_output_exponential_graphs(exp):
    return {
            'yhat': exp['yhat'].tolist(),
            'ds': exp['ds'],
            'days_to_train': exp['days'],
            'smoothing': exp['smoothing'],
            'slope': exp['slope'],
            'state': exp['state'],
            }

# {
            # 'yhat': exp['yhat'].tolist(),
            # 'ds': exp['ds'],
            # 'metadata': {
            #     'state': exp['state'],
            #     'days': exp['days'],
            #     'smoothing': exp['smoothing'],
            #     'slope': exp['slope']
            # }
            # }

if __name__ == "__main__":
    
    models, models_metadata, models_details, models_names= load_models()

