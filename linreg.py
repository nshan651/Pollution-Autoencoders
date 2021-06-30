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


def linreg(dfx, y, dims, component):
    '''
    Autoencoder model
    
    @params:
        dfx: DataFrame of normalized x values
        y: Predicted value
        dims: Number of dimensions
        component: Name of gas or particulate
    '''

    regr = LinearRegression()
    variance_list = []
    r2_list = []
    num_of_comp = list(range(2,dims+1))
    for i in num_of_comp:
        print('---------- Autoencoder dim {} for {} ----------'.format(i, component))
        # Training and test splits
        train, dev, train_labels, dev_labels = train_test_split(dfx, y, test_size=0.20, random_state=40)
        train, test, train_labels, test_labels = train_test_split(train, train_labels, test_size=0.20, random_state=40)
        input_data = Input(shape=(191,))

        X_train = train.loc[:, train.columns]
        X_test = test.loc[:, test.columns]
        X_dev = dev.loc[:, dev.columns]

       
        # Variance score explanation
        regr.fit(train, train_labels)

        #encoded_data_test = encoder.predict(x_test)
        Y_pred = regr.predict(X_test) # << Pass x_test to get predicitons for original uncompressed

        # Variance and r2 scores for the regression model
        variance = regr.score(X_test, test_labels) # << pass in x test for uncompressed values
        r2_val = r2_score(test_labels, Y_pred)
        print ('Variance score: %.2f' % variance)
        print ('R Square', r2_val)

        # Add r2/variance to lists
        variance_list.append(variance)
        r2_list.append(r2_val)        

        return (variance_list, r2_list, X_test)