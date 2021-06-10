import requests
import json
import pandas as pd
import config
import matplotlib.pyplot as plt
import datetime
import asyncio
from csv import writer


def gen_point_data(name, lat, lon, t_start, t_end):
    '''
    Generate particulate PM2.5 data for a lat/lon point over a set time period and write to csv

    @params: 
        name: name of location
        lat, lon: latitude and longitude in decimal degrees
        t_start, t_end: starting and ending epoch in Unix time
    '''

    # Connect to endpoint and load data
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
    df_size = len(ls)
    
    # Take daily averages of PM2.5 particulate and add to list
    pm_list = []
    pm_count = 0
    for i in range(df_size):
        pm_count+=ls[i]['components']['pm2_5']
        # Average for each day
        if (i%24 == 0):   
            pm_daily = round(pm_count/24, 5)
            pm_list.append(pm_daily)
            pm_count = 0

    return pm_list


def batch_request(city_df, col_names):
    ''' 
    Places a batch request for PM2.5 levels on every minute 

    @params:
        col_names: list of columns including city_name, lat/lon, and PM2.5 features
        city_df: data frame containing city information
    '''

    # Write header
    with open('C:\\github_repos\\Universal-Embeddings\\data\\geocoded-cities-master.csv', 'w', newline='') as f_open:
        writer_obj = writer(f_open)
        writer_obj.writerow(col_names)
        f_open.close()

    # Set up event loop
    loop = asyncio.get_event_loop()
    
    try:
        loop.run_until_complete(request_buffer(city_df))
    except KeyboardInterrupt:
        pass
    finally:
        print('Exiting loop')
        loop.close()

    
async def request_buffer(city_df):
    ''' Async function to buffer API requests to 60/min 
        @params:
            city_df: data frame containing city information
    '''

    curr_index = 0
    df_size = len(city_df)
    # Run batch request every minute
    while (curr_index < curr_index+60):
        print('\nPlacing batch request...')
        # Write entry to file
        with open('C:\\github_repos\\Universal-Embeddings\\data\\geocoded-cities-master.csv', 'a', newline='') as f_open:
            writer_obj = writer(f_open)
            # Loop through 60 cities from current index
            for i in range(curr_index, curr_index+60):
                city_name = city_df.iloc[i][0]
                city_lat = city_df.iloc[i][1]
                city_lon = city_df.iloc[i][2]
                city_info = [city_name, city_lat, city_lon]

                # Retrieve particulate list; write row to csv
                entry = gen_point_data(name=city_name, lat=city_lat, lon=city_lon, t_start=T_START, t_end=T_END)
                city_info+=entry
                writer_obj.writerow(city_info)
                
        f_open.close()
    
        print('Request placed, index updated to: ', curr_index)

        # Check if current position has reached the end of file
        if curr_index < df_size:
            # Increment index to next batch of 60
            curr_index+=60
            await asyncio.sleep(60)
        else:
            # Otherwise, return from event loop
            break
    
    # Finish running
    return 0
      
    
### Retrieve data for list of cities ###
city_df = pd.read_csv(filepath_or_buffer='C:\\github_repos\\Universal-Embeddings\\data\\city_lat_lon.csv')
city_count = 5 # Actual: len(city_df)

# Start and ending times. Testing for Dec 2020
T_START = 1606456800
T_END = 1623128400

# Derive number of entries from start and end
# Change in epoch to number of hours gets us total entries
num_entries = int((T_END - T_START) / 86400)
time_step = T_START

# Get list of column names based on the number of entries (each hour of data will be one column)
col_names = ['city', 'lat', 'lon']
for i in range(num_entries):
    # Convert daily interval to human-readable
    timedate = datetime.datetime.fromtimestamp(time_step)
    time_string = timedate.strftime('pm25_%Y_%m_%d')
    # Increment time_step
    time_step+=86400
    # Append col to list
    col_names.append(time_string)


### Init batch request ###
batch_request(city_df=city_df, col_names=col_names)







