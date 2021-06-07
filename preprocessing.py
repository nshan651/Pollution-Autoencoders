import requests
import json
import pandas as pd
import config
import matplotlib.pyplot as plt


def gen_point_data(lat, lon, t_start, t_end):
    '''
    Generate pollution data for a lat/lon point over a set time period and write to csv

    @params: 
        lat, lon: latitude and longitude in decimal degrees
        t_start, t_end: starting and ending epoch in Unix time
    '''

    # Connect to endpoint and load data
    # test_endpoint = 'http://api.openweathermap.org/data/2.5/air_pollution/history?lat=42&lon=-92&start=1606223802&end=1606482999&appid={}'.format(config.OPEN_WEATHER_KEY)
    endpoint = 'http://api.openweathermap.org/data/2.5/air_pollution/history?lat={LAT}&lon={LON}&start={START}&end={END}&appid={KEY}'.format(
        LAT=lat, 
        LON=lon,
        START=t_start,
        END=t_end,
        KEY=config.OPEN_WEATHER_KEY
    )
    page = requests.get(url=endpoint)
    content = json.loads(page.content)
    df = pd.json_normalize(content)

    # List all records
    ls = df['list'][0]

    # Number of records
    df_size = len(ls)

    # Times, aqi, gas lists
    epoch_list, aqi_list, gas_list = ([] for i in range(3))

    for i in range(df_size):
        # Extract each gas dict entry and add to list
        gas_list.append(ls[i]['components'])
        # Extract corresponding time and aqi
        epoch_list.append(ls[i]['dt'])
        aqi_list.append(ls[i]['main']['aqi'])

    # Append values to new df
    result = pd.DataFrame(data=gas_list, columns=['co', 'no', 'no2', 'o3', 'so2', 'pm2_5', 'pm10', 'nh3'], index=pd.RangeIndex(0, df_size))
    result.insert(loc=0, column='dt', value=epoch_list)
    result.insert(loc=1, column='aqi', value=aqi_list)
    print(result)

    # To csv
    result.to_csv(path_or_buf='C:\\github_repos\\Universal-Embeddings\\data\\test_data.csv', index=None)

