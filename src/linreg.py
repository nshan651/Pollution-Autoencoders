import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import operator
from numpy.core.numeric import NaN
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import Normalizer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import r2_score


def linreg(X_train, X_test, Y_train, Y_test, component, folds=0, dev=False):
    '''
    Simple linear regression

    @params:
        dfx: DataFrame of normalized x values
        y: Predicted value
        component: Name of gas or particulate
    '''

    regr = LinearRegression()
    
    input_data = Input(shape=(191,)) # <- Not sure if needed
    
    # Linear regression on validation/train set
    if dev == True:
        print(f'Linear regression validation -- fold {folds} -- component {component}')
        train_reg, dev_reg, train_labels_reg, dev_labels_reg = train_test_split(X_train, Y_train, test_size=0.33, random_state=42)
        regr.fit(train_reg, train_labels_reg)
        res_sum_square= np.mean((regr.predict(dev_reg) - dev_labels_reg ** 2))
        var_val=regr.score(dev_reg, dev_labels_reg)
        Y_test_predict = regr.predict(dev_reg)
        r2_val=r2_score(dev_labels_reg, Y_test_predict)
        return (regr.intercept_, regr.coef_, res_sum_square, var_val, r2_val)

    # Linear regression on test set
    else:
        print(f'Linear regression test for {component}')
        regr.fit(X_train, Y_train)
        res_sum_square= np.mean((regr.predict(X_test) - Y_test ** 2))
        var_val=regr.score(X_test, Y_test)
        Y_test_predict = regr.predict(X_test)
        r2_val=r2_score(Y_test, Y_test_predict)
        return (var_val, r2_val, X_train)


def xcross(x, y, splits, component, dims=1):
    '''
    Run the linear regression for a single gas
    using k-fold cross validation strategy

    @params:
        x: x input values 
        y: Response values
        splits: Number of folds for k-fold cross val
        component_names: Gas/particulate name list
        dims: number of dimensions to test over. Default is 1
    '''

    print(f'---------- Beginning Linear Regression for {component} ----------')

    # k-fold cross validation
    kfold = KFold(n_splits=splits, shuffle=True, random_state=10)
    folds=0

    # Number of features to compare
    num_of_dims = np.arange(2, dims+1, 1) if dims!=1 else [1]

    # Train/test and metrics for current component's set of data
    train_test = {}
    metrics={}
    # Contains all train_test splits/metrics in REG for n sets of splits
    train_test_dict = {}
    metrics_dict = {}
    metrics_dict_set = {}

    # Loop through train/test data and save the best data with highest R2 scores
    for training_index, test_index in kfold.split(x):
        # Split X and Y train and test data
        X_train, X_test = x[training_index, :], x[test_index, :]
        Y_train, Y_test = y[training_index], y[test_index]
        folds+=1
        # Update dict with train/test values
        train_test['X_train'] = X_train
        train_test['X_test'] = X_test
        train_test['Y_train'] = Y_train
        train_test['Y_test'] = Y_test
        # Save best sets of train/test data that have high R2 scores
        train_test_dict[folds] = train_test.copy()
        # Train regression and save list of metrics
        for dim in num_of_dims:
            # Train regression model and save a list of metrics 
            # TODO: change this to
            model_intercept, model_coef, res_sum_square, variance_score, Rsquare = linreg(X_train, X_test, Y_train, Y_test, component, folds, dev=True)
            # Create metrics list for current comparison
            metrics['model_intercept']=model_intercept
            metrics['model_coef']=model_coef
            metrics['res_sum_square']=res_sum_square
            metrics['variance_score']=variance_score
            metrics['Rsquare']=Rsquare
            metrics_dict[dim] = metrics.copy()
        # Save each metrics comparison for later
        metrics_dict_set[folds] = metrics_dict.copy()
        #print(metrics_dict_set)

    # Calculate the best and worst R2 scores for each dimension 
    # Set best and worst R2 to the first appearance of R2
    if (dims==1):
        best_r2=worst_r2=metrics_dict_set[1][1]['Rsquare']
    else:
        best_r2=worst_r2=metrics_dict_set[1][2]['Rsquare']
    best_idx=worst_idx=0
    
    # Best and worst R2 scores, [('position','score')]
    best=worst=[]

    for fold in metrics_dict_set:
        for dim in metrics_dict_set[fold]:
            R2 = metrics_dict_set[fold][dim]['Rsquare']
            if R2 >= best_r2:
                best_r2 = R2
                best_idx = fold
            elif R2 <= worst_r2:
                worst_r2 = R2
                worst_idx = fold
            print(f' dim {dim} best: {best_r2}')
            print(f'dim {dim} worst: {worst_r2}')
    # Note: Not sure if block below is needed for cross val???       
    # Obtain the values based off of the indexes of the best and worst R2 scores
    X_tt_best=X_tn_best=Y_tt_best=Y_tn_best=X_tt_worst=X_tn_worst=Y_tt_worst=Y_tn_worst=train_test_dict[1]['X_test']
    for i in train_test_dict:
        if i == best_idx:
            for set_type in train_test_dict[best_idx]:
                if 'X_test'==set_type:
                    X_tt_best=train_test_dict[best_idx][set_type]
                elif 'X_train' == set_type:
                    X_tn_best=train_test_dict[best_idx][set_type]
                elif 'Y_test' == set_type:
                    Y_tt_best=train_test_dict[best_idx][set_type]
                elif 'Y_train' == set_type:
                    Y_tn_best=train_test_dict[best_idx][set_type]
        elif i == worst_idx:
            for set_type in train_test_dict[worst_idx]:
                if 'X_test'==set_type:
                    X_tt_worst=train_test_dict[worst_idx][set_type]
                elif 'X_train' ==set_type:
                    X_tn_worst=train_test_dict[worst_idx][set_type]
                elif 'Y_test' ==set_type:
                    Y_tt_worst=train_test_dict[worst_idx][set_type]
                elif 'Y_train'==set_type:
                    Y_tn_worst=train_test_dict[worst_idx][set_type]

   

### RUN ###

#COMPONENT_NAMES = ['co', 'no', 'no2', 'o3', 'so2', 'pm2_5', 'pm10', 'nh3']
COMPONENT = 'co'
DIMS = 5
SPLITS = 5
# Open normalized data
dfx = pd.read_csv(f"{os.environ['HOME']}/github_repos/Pollution-Autoencoders/data/data_norm/{COMPONENT}_data_norm.csv")
# y value list using last day of 7-month data
dfy = pd.read_csv(f"{os.environ['HOME']}/github_repos/Pollution-Autoencoders/data/data_clean/{COMPONENT}_data_clean.csv")
x = dfx.values
y = dfy.loc[:, [f'{COMPONENT}_2021_06_06']].values

xcross(x, y, SPLITS, COMPONENT)

