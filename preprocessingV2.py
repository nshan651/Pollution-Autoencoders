import requests
import json
import pandas as pd
import config
import matplotlib.pyplot as plt
import datetime
import asyncio
from csv import writer

# Version 2 of preprocessing
# generalizes and obtains lists for each particulate/gas component


def gen_daily_data(name, lat, lon, t_start, t_end, component_names):
    '''
    Generate data for a given gas/particulate for a given city over a set time period and write to csv

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
    
    # Hourly time series list of each entry; daily total of each entry
    ts_list = []
    component_dict = {}
    total = 0

    # Find the daily averages of each component; merge into component list
    for component in component_names:
        for i in range(df_size):
            total+=ls[i]['components'][component]
            # Average for each day
            if (i%24 == 0):   
                daily_value = round(total/24, 5)
                ts_list.append(daily_value)
                total = 0
        # Each time series list goes into a new dict entry
        component_dict[component] = ts_list
        # Clear list for next iteration
        ts_list = []
    return component_dict


def gen_cols(t_start, t_end, component):
    ''' 
    Get list of column names based on the number of entries (each daily average of data will be one column)
    Derive number of entries from start and end

    @params:
        t_start, t_end: start and end times in unix time
        element: string to represent the gas/particulate name to use
    '''

    # Change in epoch to number of hours gets us total entries
    num_entries = int((t_end - t_start) / 86400)
    time_step = T_START
    col_names = ['city', 'lat', 'lon']

    for i in range(num_entries):
        # Convert daily interval to human-readable
        timedate = datetime.datetime.fromtimestamp(time_step)
        time_string = timedate.strftime('{COMPONENT}_%Y_%m_%d'.format(COMPONENT=component))
        # Increment time_step
        time_step+=86400
        # Append col to list
        col_names.append(time_string)

    return col_names


def batch_request(city_df, component_names):
    ''' 
    Places a batch request for gas/particulate levels on every minute 

    @params:
        col_names: list of columns including city_name, lat/lon, and PM2.5 features
        city_df: data frame containing city information
    '''
    
    # Set up event loop
    loop = asyncio.get_event_loop()
    
    try:
        loop.run_until_complete(request_buffer(city_df, component_names))
    except KeyboardInterrupt:
        pass
    finally:
        print('Exiting loop')
        loop.close()
    
    
async def request_buffer(city_df, component_names):
    ''' Async function to buffer API requests to 60/min 
        @params:
            city_df: data frame containing city information
    '''

    curr_index = 0
    df_size = len(city_df)
    # Run batch request every minute
    while (curr_index <= curr_index+60):
        print('\nPlacing batch request...')

        # Loop through 60 cities from current index
        for i in range(curr_index, curr_index+60):
            city_name = city_df.iloc[i][0]
            city_lat = city_df.iloc[i][1]
            city_lon = city_df.iloc[i][2]
            city_info = [city_name]

            # Retrieve particulate dict
            entry = gen_daily_data(name=city_name, lat=city_lat, lon=city_lon, t_start=T_START, t_end=T_END, component_names=component_names)
            
            # Cycle through each component and write to file
            for component in component_names:
                # Write entry to file
                file_name = 'C:\\github_repos\\Pollution-Autoencoders\\data\\gases\\{}.csv'.format(component)
                with open(file_name, 'a', newline='') as f_open:
                    writer_obj = writer(f_open)
                    city_info+=entry[component]
                    writer_obj.writerow(city_info)
                f_open.close()
    
        print('Request placed, index updated to: ', curr_index)

        # Check if current position has reached the end of file
        if curr_index < df_size:
            # Increment index to next batch of 60
            curr_index+=60
            await asyncio.sleep(90)
        else:
            # Otherwise, return from event loop
            break
    
    # Finish running
    return 0
      
    
### Retrieve data for list of cities ###
city_df = pd.read_csv(filepath_or_buffer='C:\\github_repos\\Pollution-Autoencoders\\data\\city_lat_lon.csv')
city_count = len(city_df)

# Start and ending times. Testing for Dec 2020
T_START = 1606456800
T_END = 1623128400
COMPONENT_NAMES = ['co', 'no', 'no2', 'o3', 'so2', 'pm2_5', 'pm10', 'nh3']

# Generate column names
col_names_dict = {}

# Write headers for each component
for component in COMPONENT_NAMES:
    # generate column names
    col_names_dict[component] = list(gen_cols(T_START, T_END, component))
    file_name = 'C:\\github_repos\\Pollution-Autoencoders\\data\\gases\\{COMPONENT}.csv'.format(COMPONENT=component)
    with open(file_name, 'w', newline='') as f_open:
        writer_obj = writer(f_open)
        writer_obj.writerow(col_names_dict[component][:])
        f_open.close()

### Init batch request ###
batch_request(city_df=city_df, component_names=COMPONENT_NAMES)