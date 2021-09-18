import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


def regression(train_data, test_data, train_labels, test_labels):
    ''' 
    Perform a linear regression 
    
    @params:
        train_data: Encoded data used to fit the regression
        test_data: Encoded data used for prediction
        train_labels, test_labels: testing/training labels
    @return:
        variance, r2: The variance/r2 scores of the encoded test set
    '''    
    
    # Define the regression model
    regr = LinearRegression()

    #print('size of trained data', len(train_data))
    #print('size of trained labels', len(train_labels))
    # Fit the regression model with the encoded training data
    regr.fit(train_data, train_labels)
    
    Y_pred = regr.predict(test_data) 
    
    # Variance and r2 scores for the regression model
    variance = regr.score(test_data, test_labels) 
    r2 = r2_score(test_labels, Y_pred)    
    print ('Variance score: %.2f' % variance)
    print ('R Square: %.2f' % r2)

    return variance, r2

