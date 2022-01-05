import pandas as pd
import googlemaps
import urllib.request
import json
import keys
#import dotenv

# Reverse GeoCoding to populate data with state/country codes
# 40.714224, -73.961452
def rev_geo(lat, lon, key):
    REVGEO_BASE_URL = 'https://maps.googleapis.com/maps/api/geocode/json'
    URL_PARAMS = f'latlng={lat},{lon}&key={key}'
    
    url = REVGEO_BASE_URL + "?" + URL_PARAMS
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
    return state_code, country_code


def append_labels():
    component_names = ['co', 'no', 'no2', 'o3', 'so2', 'pm2_5', 'pm10', 'nh3']
    data_labels = pd.read_csv('/home/nick/Downloads/revised_data/data_labels.csv')
    for component in component_names:
        f_name = f'/home/nick/github_repos/Pollution-Autoencoders/data/data_clean/{component}_data_clean.csv'
        output_file = f'/home/nick/Downloads/revised_data/{component}_test.csv'
        df = pd.read_csv(f_name)
        df.insert(1, 'state', data_labels['state'])
        df.insert(2, 'country', data_labels['country'])
        df.to_csv(output_file, index=False)


append_labels()
'''
KEY = keys.GMAPS_KEY

cities = []
states = []
countries = []

df = pd.read_csv(f'./data/data_clean/co_data_clean.csv')
count = 0
for city, lat, lon in zip(df["city"], df["lat"], df["lon"]):
    print(f'{city}, {lat} , {lon}')
    
    state_code, country_code = rev_geo(lat, lon, KEY)
    states.append(state_code)
    countries.append(country_code)    
    cities.append(city)

# Add new columns to the df
label_dict = {'city' : cities, 'state' : states, 'country' : countries}
labeled_data = pd.DataFrame(data=label_dict)
labeled_data.to_csv(f'/home/nick/Downloads/revised_data/data_labels.csv', index=False)
'''
