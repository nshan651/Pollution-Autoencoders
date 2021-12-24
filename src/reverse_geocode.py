import pandas as pd
import googlemaps
import urllib.request
import json
#import dotenv

# Reverse GeoCoding to populate data with state/country codes
# 40.714224, -73.961452
def rev_geo(lat, lon, key):
    REVGEO_BASE_URL = 'https://maps.googleapis.com/maps/api/geocode/json'
    URL_PARAMS = f'latlng={lat},{lon}&key={key}'
    
    url = REVGEO_BASE_URL + "?" + URL_PARAMS
    #print(f'url is {url}')
    # Read the contents of the generated url and decode the result
    with urllib.request.urlopen(url) as f:
        response = json.loads(f.read().decode())
    status = response["status"]
    if status == "OK":
        result = response["results"][0]['address_components']
        #print(result)

        state_code = country_code = 'NULL'
        # Parse the response object for administrative_area_level_1 and country types
        for r in result:
            if r['types'][0] == 'administrative_area_level_1':
                state_code = r['short_name']
            if r['types'][0] == 'country':
                country_code = r['short_name']
    else:
        print(status)
        print(response['error_message'])
        rev_geo_data = None
    return (state_code, country_code)

KEY = 'AIzaSyCZr4o4NiY5tKHlWNVTh9ExeiKQ_O35UMc'
#LAT = 30.2671500
#LON = -97.7430600
component_names = ['co', 'no', 'no2', 'o3', 'so2', 'pm2_5', 'pm10', 'nh3']
states = countries = []

for component in component_names:
    df = pd.read_csv(f'./data/data_clean/{component}_data_clean.csv')
    for lat, lon in zip(df['lat'], df['lon']):
        #print(f'{lat} , {lon}')
        state_code, country_code = rev_geo(lat, lon, KEY)
        states.append(state_code)
        countries.append(country_code)
        
    # Add new columns to the df
    df.insert(loc = 1, column = 'state', value = states)
    df.insert(loc = 2, column = 'country', value = countries)
    df.to_csv(f'/home/nick/Downloads/revised_data/{component}_data_clean.csv')
#print(data)