import requests
import json
import pandas as pd
import config

### API connection/data retrieval test ###

# Connect to endpoint and load into df
endpoint = 'http://api.openweathermap.org/data/2.5/air_pollution/history?lat=42&lon=-92&start=1606223802&end=1606482999&appid={}'.format(config.OPEN_WEATHER_KEY)
page = requests.get(url=endpoint)
content = json.loads(page.content)
df = pd.json_normalize(content)

# List of all records
ls = df['list'][0]

# Number of records
df_size = len(ls)

# First record (Jan 24, 2020)
r1 = ls[0]

# Epoch timestamps
epoch = r1['dt']

# Air quality index
aqi = r1['main']['aqi'] 

# Polluting gases
co = r1['components']['co']
no = r1['components']['no']








