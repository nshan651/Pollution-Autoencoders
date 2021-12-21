import autoencoder as ae
import numpy as np
import itertools
import pandas as pd
import os
import sys
from sklearn.model_selection import train_test_split


def main(*argv):
    # Starting dimensions
    dims = 190
    # K-fold folds
    folds = 5
    # Grid search params
    lr = [0.0001, 0.001, 0.01, 0.1]
    batch = [32, 64, 128, 256]
    epochs = [10, 50, 75, 100, 150]
    # Param vector
    iter_dims = []
    # List of key params to test
    key_params = list(itertools.product(lr,batch,epochs))
    ### Input files ###
    # Open normalized data and dependent, non-normalized data
    dfx = pd.read_csv(f"{os.environ['HOME']}/github_repos/Pollution-Autoencoders/data/data_norm/co_data_norm.csv")
    dfy = pd.read_csv(f"{os.environ['HOME']}/github_repos/Pollution-Autoencoders/data/data_clean/co_data_clean.csv")
    # City names to append
    cities = dfy['city'].values
    # Grid params to create model with
    param_grid = pd.read_csv(f'/home/nick/github_repos/Pollution-Autoencoders/data/grid_params/hyperparams.csv')
    # Name of the component gas
    component = 'no2'
    file_name=''
    # Set x as the normalized values, y (non-normalized) as the daily average of final day
    X = dfx.values
    Y = dfy.loc[:, ['co_2021_06_06']].values
    # Split into train/test data
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=40)
    
    # Arguments
    for args in sys.argv:   
        if args == "-k_1_5" or args == "--first-five":
            iter_dims = np.arange(1,6,1)
            file_name = f'/home/nick/github_repos/Pollution-Autoencoders/data/grid_params/{component}_grid_params_1_5.csv'
        elif args == "-k_6_9" or args == "--six-to-nine":
            iter_dims = np.arange(6,10,1)
            file_name = f'/home/nick/github_repos/Pollution-Autoencoders/data/grid_params/{component}_grid_params_6_9.csv'
        elif args == "-k_10_60" or args == "--ten-to-fifty":
            iter_dims = np.arange(10,51,10)
            file_name = f'/home/nick/github_repos/Pollution-Autoencoders/data/grid_params/{component}_grid_params_10_50.csv'
        elif args == "-k_70_120" or args == "--seventy-to-onetwenty":
            iter_dims = np.arange(70,121,10)
            file_name = f'/home/nick/github_repos/Pollution-Autoencoders/data/grid_params/{component}_grid_params_50_100.csv'
        elif args == "-k_all" or args == "--run-all":
            iter_dims = np.append(np.arange(1, 10, 1), np.arange(10, 120, 10))
            file_name = f'/home/nick/github_repos/Pollution-Autoencoders/data/grid_params/{component}_grid_params_all.csv'

    # Perform grid search
    ae.grid_search(
        file_name=file_name,
        x=X,
        y=Y,
        folds=folds,
        component=component,
        iter_dims=iter_dims,
        key_params=key_params
    )

if __name__ == '__main__':
    main()
