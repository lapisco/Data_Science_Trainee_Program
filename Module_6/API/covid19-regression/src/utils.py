from datetime import datetime, timedelta
from os import path
import numpy as np
import pandas as pd

import requests, json


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

    
def format_date(date_list, date_string_input='%m-%d-%y'):
    dates = [datetime.strptime(x, date_string_input) for x in date_list]
    dates_tormated = [x.strftime('%a, %d %b %Y %H:%m:%S') for x in dates]

    return dates_tormated


# def porra(df):
#     # new_df = df.iloc[-16:].copy()
#     # new_df = df.copy()
#     new_df = df
#     new_df.reset_index(inplace=True)
#     new_df.drop(labels=['cured', 'deaths', 'refuses', 'suspects'], axis=1, inplace=True)
#     new_df.rename(columns={'data': 'ds', 'cases': 'y'}, inplace=True)

#     # new_df['y']=new_df['y']*0.05

#     return new_df


def porra(df):
    df_new = df.loc[df['cases'] != 0]
    # df_new.drop(labels=['cured', 'deaths', 'refuses', 'suspects'], axis=1, inplace=True)
    df_new.drop(labels=LABELS_TO_DROP + LABELS_TO_DROP_OLD, axis=1, inplace=True)
    # df_new = df_new['cases'].reset_index().drop(labels=['index'], axis=1)
    df_new.rename(columns={'cases': 'y'}, inplace=True)

    return df_new


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


def check_inputs(X, y):
    if X.shape[0] < y.shape[0]:
        diff = np.abs(X.shape[0] - y.shape[0])
        y_new = y[diff:]
        X_new = X
    elif X.shape[0] > y.shape[0]:
        diff = np.abs(X.shape[0] - y.shape[0])
        X_new = X[diff:]
        y_new = y
    else:
        X_new = X
        y_new = y

    return X_new, y_new


if __name__ == "__main__":
    
    
    print(format_date(DATE_LIST))


