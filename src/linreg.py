import numpy as np
from numpy.core.numeric import NaN
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import Normalizer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


def linreg(dfx, y, component):
    '''
    Simple linear regression on uncompressed values
    
    @params:
        dfx: DataFrame of normalized x values
        y: Predicted value
        component: Name of gas or particulate
    '''

    regr = LinearRegression()
    variance_list = []
    r2_list = []
    
    # Training and test splits
    train, dev, train_labels, dev_labels = train_test_split(dfx, y, test_size=0.20, random_state=40)
    train, test, train_labels, test_labels = train_test_split(train, train_labels, test_size=0.20, random_state=40)
    input_data = Input(shape=(191,))

    X_train = train.loc[:, train.columns]
    X_test = test.loc[:, test.columns]
    X_dev = dev.loc[:, dev.columns]
    
    # Variance score explanation
    regr.fit(train, train_labels)

    Y_pred = regr.predict(X_test) # << Pass x_test to get predicitons for original uncompressed

    # Variance and r2 scores for the regression model
    variance = regr.score(X_test, test_labels) # << pass in x test for uncompressed values
    r2_val = r2_score(test_labels, Y_pred)
    print ('Variance score: %.2f' % variance)
    print ('R Square', r2_val)

    # Add r2/variance to lists
    variance_list.append(variance)
    r2_list.append(r2_val)        

    print('variance:',variance_list)
    print('r2: ', r2_list)
    print('x test: ', X_test)
    return (variance_list, r2_list, X_test)


def linreg_run(component_names):
    '''
    Run the linear regression for each component gas

    @params:
        component_names: Gas/particulate name list
    '''
    for component in component_names:
        print('---------- Running Linear Regression for {} ----------'.format(component))
        df = pd.read_csv('/home/nicks/github_repos/Pollution-Autoencoders/data/data_clean/{}_data_clean.csv'.format(component))
        
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

        # Perform the regression        
        variance, r2, X_train = linreg(dfx, y, component)

### RUN ###

#COMPONENT_NAMES = ['co', 'no', 'no2', 'o3', 'so2', 'pm2_5', 'pm10', 'nh3']
COMPONENT_NAMES = ['no2']
COLORS_LIST = ['tab:blue', 'tab:green', 'tab:orange', 'tab:red', 'tab:purple', 'tab:cyan', 'tab:olive', 'tab:pink']


linreg_run(COMPONENT_NAMES)

