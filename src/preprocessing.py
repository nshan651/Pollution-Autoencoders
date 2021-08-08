import requests
import json
import pandas as pd
import datetime
import asyncio
import os
from sklearn.preprocessing import Normalizer
import logging
import sys
from dotenv import load_dotenv
from csv import writer


load_dotenv()   # Load environment variables

def gen_daily_data(name, lat, lon, t_start, t_end, component_names):
    '''
    Generate data for a given gas/particulate for a given city over a set time period and write to csv

    @params: 
        name: Name of location
        lat, lon: Latitude and longitude in decimal degrees
        t_start, t_end: Starting and ending epoch in Unix time
    '''
    
    # Connect to endpoint and load data
    endpoint = 'http://api.openweathermap.org/data/2.5/air_pollution/history?lat={LAT}&lon={LON}&start={START}&end={END}&appid={KEY}'.format(
        LAT=lat, 
        LON=lon,
        START=t_start,
        END=t_end,
        KEY=os.getenv('OPEN_WEATHER_KEY')
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


def unit_norm(file_in, file_out, component_names, dims):
    '''
    Unit Normalize the gas data on a 0-1 scale and save to file

    @params:
        file_in: File containing input data
        file_out: Destination to output unit normalized data
        component_names: List of component gases
        dims: Specify number of dimensions
    '''

    for i, component in enumerate(component_names):
        print('---------- Normalizing component {} ----------'.format(component))
        try:
            df = pd.read_csv(file_in[i])
        except:
            logging.error('Unit normalization failed, input file not recognized')
            sys.exit(1)
        # Features list and removal of city, lat, lon
        features = list(df.columns.values)
        del features[:3] 
        del features[-1]

        # y value list using last day of 7-month data
        y = df.loc[:, ['{}_2021_06_06'.format(component)]].values
        # Normalize x values; save in data frame
        x = df.loc[:, features].values
        x = Normalizer().fit_transform(x)
        dfx = pd.DataFrame(x)

        # Create columns names for each dimension 
        norm_labels = ['dim_{}'.format(i) for i in range(1, dims+2)]
        dfx.columns = norm_labels

        # Write to file
        write_data = pd.DataFrame(data=dfx)
        write_data.to_csv(path_or_buf=file_out[i], index=False)
        

def gen_cols(t_start, t_end, component):
    ''' 
    Get list of column names based on the number of entries (each daily average of data will be one column)
    Derive number of entries from start and end

    @params:
        t_start, t_end: Start and end times in unix time
        element: String to represent the gas/particulate name to use
    '''

    # Change in epoch to number of hours gets us total entries
    num_entries = int((t_end - t_start) / 86400)
    time_step = t_start
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


def batch_request(f_name, t_start, t_end, city_df, component_names):
    ''' 
    Places a batch request for gas/particulate levels on every minute 

    @params:
        f_name: Name of files to write to
        col_names: List of columns including city_name, lat/lon, and PM2.5 features
        city_df: Data frame containing city information
    '''
    
    # Set up event loop
    loop = asyncio.get_event_loop()
    
    try:
        loop.run_until_complete(request_buffer(f_name, t_start, t_end, city_df, component_names))
    except KeyboardInterrupt:
        pass
    finally:
        print('Exiting loop')
        loop.close()
    
    
async def request_buffer(f_name, t_start, t_end, city_df, component_names):
    ''' Async function to buffer API requests to 60/min 
        @params:
            f_name: name of files to write to
            city_df: Data frame containing city information
            component names: List of component gases
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
            entry = gen_daily_data(name=city_name, lat=city_lat, lon=city_lon, t_start=t_start, t_end=t_end, component_names=component_names)
            
            # Cycle through each component and write to file
            for i, component in enumerate(component_names):
                # Write entry to file
                with open(f_name[i], 'a', newline='') as f_open:
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
    

def main():
    ''' 
    Main function to pull data csv data from OpenWeather

    Function call order: 
        main() > batch_request() > request_buffer() > gen_daily_data()
    
    '''

    # Retrieve data for list of cities 
    city_df = pd.read_csv(filepath_or_buffer=f"{os.environ['HOME']}/github_repos/Pollution-Autoencoders/data/other/city_lat_lon.csv")
    city_count = len(city_df)
    
    # Start and ending times
    # Defaults are 2020-11-27 to 2021-06-06
    T_START = 1606456800
    T_END = 1623128400
    # Component gases
    COMPONENT_NAMES = ['co', 'no', 'no2', 'o3', 'so2', 'pm2_5', 'pm10', 'nh3']
    # List of file names to write to
    #DATA_OUT = [f"{os.environ['HOME']}/github_repos/Pollution-Autoencoders/{component}_test.csv" for component in COMPONENT_NAMES]
    DATA_OUT = [f"{os.environ['HOME']}/github_repos/Pollution-Autoencoders/data/data_clean/{component}_data_clean.csv" for component in COMPONENT_NAMES]
    # Output location of normalized data
    NORM_OUT = [f"{os.environ['HOME']}/github_repos/Pollution-Autoencoders/{component}_norm_test.csv" for component in COMPONENT_NAMES]
    # Dimensions to use
    DIMS = 190
    # Column names
    col_names_dict = {}
    print(DATA_OUT)
    '''
    # Write headers for each component
    for i, component in enumerate(COMPONENT_NAMES):
        # generate column names
        col_names_dict[component] = list(gen_cols(T_START, T_END, component))
        with open(DATA_OUT[i], 'w', newline='') as f_open:
            writer_obj = writer(f_open)
            writer_obj.writerow(col_names_dict[component][:])
            f_open.close()
    '''
    ### Function calls ###
    #batch_request(DATA_OUT, T_START, T_END, city_df, COMPONENT_NAMES)
    unit_norm(DATA_OUT, NORM_OUT, COMPONENT_NAMES, DIMS)
    

if __name__=='__main__':
    main()

