import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
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
    Simple linear regression on uncompressed values

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


def linreg_run(component_names, dims):
    '''
    Run the linear regression for each component gas
    Uses k-fold cross validation strategy

    @params:
        component_names: Gas/particulate name list
    '''

    for component in component_names:
        print('---------- Beginning Linear Regression for {} ----------'.format(component))
        # Open normalized data
        dfx = pd.read_csv(f"{os.environ['HOME']}/github_repos/Pollution-Autoencoders/data/data_norm/{component}_data_norm.csv")
        # y value list using last day of 7-month data
        dfy = pd.read_csv(f"{os.environ['HOME']}/github_repos/Pollution-Autoencoders/data/data_clean/{component}_data_clean.csv")
        x = dfx.values
        
        y = dfy.loc[:, ['{}_2021_06_06'.format(component)]].values
        #y = dfx.loc[:, ['dim_191']].values

        # k-fold cross validation
        kfold = KFold(n_splits=5, shuffle=True)
        folds=0
        # Train/test and metrics for current component's set of data
        train_test = {}
        metrics={}
        # Contains all train_test splits/metrics in REG for n components and 
        # m sets of splits
        train_test_dict = {}
        metrics_dict = {}

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
            
            # Train regression model and save a list of metrics 
            model_intercept, model_coef, res_sum_square, variance_score, Rsquare = linreg(X_train, X_test, Y_train, Y_test, component, folds, dev=True)
            # Create metrics list for current comparison
            metrics['model_intercept']=model_intercept
            metrics['model_coef']=model_coef
            metrics['res_sum_square']=res_sum_square
            metrics['variance_score']=variance_score
            metrics['Rsquare']=Rsquare
            
            # Save each metrics comparison for later
            metrics_dict[folds] = metrics.copy()
            #print(metrics_dict)
        
        # Calculate the best and worst R2 scores for each component
        best_r2, worst_r2 = metrics_dict[1]['Rsquare'], metrics_dict[1]['Rsquare']
        best_idx, worst_idx = 0, 0
        for i in metrics_dict:
            R2 = metrics_dict[i]['Rsquare']
            if R2 >= best_r2:
                best_r2 = R2
                best_idx = i
            elif R2 < worst_r2:
                worst_r2 = R2
                worst_idx = i

        # Obtain the values based off of the indexes of the best and worst R2 scores
        for i in train_test_dict:
            if i == best_idx:
                for sets in train_test_dict[best_idx]:
                    if 'X_test'==sets:
                        X_tt_best=train_test_dict[best_idx][sets]
                    elif 'X_train' ==sets:
                        X_tn_best=train_test_dict[best_idx][sets]
                    elif 'Y_test' ==sets:
                        Y_tt_best=train_test_dict[best_idx][sets]
                    elif 'Y_train'==sets:
                        Y_tn_best=train_test_dict[best_idx][sets]
            elif i == worst_idx:
                for sets in train_test_dict[worst_idx]:
                    if 'X_test'==sets:
                        X_tt_worst=train_test_dict[worst_idx][sets]
                    elif 'X_train' ==sets:
                        X_tn_worst=train_test_dict[worst_idx][sets]
                    elif 'Y_test' ==sets:
                        Y_tt_worst=train_test_dict[worst_idx][sets]
                    elif 'Y_train'==sets:
                        Y_tn_worst=train_test_dict[worst_idx][sets]
        
        ### Perform the regression for the highest and lowest scores ###   
        # Note: Sometimes the best-performing folds of training data perform worse
        # on the test set

        # For the best values 
        best_var, best_r2, best_X_vec = linreg(
            X_train=X_tn_best, 
            X_test=X_tt_best, 
            Y_train=Y_tn_best, 
            Y_test=Y_tt_best, 
            component=component)
        # For the worst values
        worst_var, worst_r2, worst_X_vec = linreg(
            X_train=X_tn_worst, 
            X_test=X_tt_worst, 
            Y_train=Y_tn_worst, 
            Y_test=Y_tt_worst, 
            component=component)

        print(f'best var: {best_var} -- best r2: {best_r2}')
        print(f'worst var: {worst_var} -- best r2: {worst_r2}')
        
### RUN ###

#COMPONENT_NAMES = ['co', 'no', 'no2', 'o3', 'so2', 'pm2_5', 'pm10', 'nh3']
COMPONENT_NAMES = ['co']
COLORS_LIST = ['tab:blue', 'tab:green', 'tab:orange', 'tab:red', 'tab:purple', 'tab:cyan', 'tab:olive', 'tab:pink']
DIMS = 190

linreg_run(COMPONENT_NAMES, DIMS)

