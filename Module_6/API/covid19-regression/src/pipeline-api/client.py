import requests
import pandas as pd
import json

from decouple import config as cfg

PORT = cfg('PORT', default=80, cast=int)

URL = 'http://localhost:{}/churn-detector/pipeline'.format(5000)

# headers = {'content-type': content_type}
# json = pd.read_json('input_data.json', orient='index')
json.loads('input_data.jsonw')
print(json)

response = requests.post(URL, json=json)
print(response.text)