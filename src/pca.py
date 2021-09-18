import pandas as pd
import numpy as np
import csv
import os
import linreg # src/linreg.py
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import Normalizer
from sklearn.cluster import KMeans

def pca(i, X_train, Y_train, component):
    '''
    Perform Linear Regression on either train/validation set or on test set for PCA

    @params:
        i: Number of components
        X_train, Y_train: Training data
        component: Gas/particulate 
    '''

    # Define the model and fit the trained embedding
    pca = PCA(n_components=i)
    encoded_X_train = pca.fit_transform(X_train)
    
    return pca, encoded_X_train


def pca_train_test(dims, x, y, folds, component):
    ''' 
    Run PCA using k-fold cross validation strategy

    @params:
        dims: Number of dimensions to train
        x, y: Independent/dependent data used for k-fold split
        folds: Number of folds to iterate through
        component: The name of the gas/particulate
    '''
    
    # Metrics file
    file_name = f'/home/nick/github_repos/Pollution-Autoencoders/data/model_metrics/pca/{component}_metrics'
    # Headers 
    test_metrics_list = ['fold', 'dim', 'variance', 'r2']
    # Write header
    with open(file_name,'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(test_metrics_list)
        f.close() 

    # k-fold cross validation; any cross-validation technique can be substituted here
    kfold = KFold(n_splits=folds, shuffle=True)
    # Number of features to compare
    num_of_comp=list(range(1,dims+1))
    fold_count=0

    print(f'---------- Beginning PCA for gas {component} ----------')
    # Loop through train/test data and save the best data with highest R2 scores
    for training_index, test_index in kfold.split(x):
        # Split X and Y train and test data
        X_train, X_test = x[training_index, :], x[test_index, :]
        Y_train, Y_test = y[training_index], y[test_index]
        fold_count+=1

        # Train PCA and save a list of metrics 
        for i in num_of_comp:
            # Train pca model
            model, encoded_train_data = pca(
                i=i, 
                X_train=X_train,  
                Y_train=Y_train, 
                component=component)
            # Create the test embedding
            encoded_test_data = model.transform(X_test)
            # Perform linear regression
            variance, r2 = linreg.regression(encoded_train_data, encoded_test_data, Y_train, Y_test)
            # Print result
            print(f'fold {fold_count} || dim {i} || variance {variance} || r2 {r2} \n')
            # Update test metrics file
            test_metrics_list = [fold_count, i, variance, r2]
            with open(file_name,'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(test_metrics_list)
                f.close() 


# TODO: FIX THIS MONSTROSITY!!!!!
def interpolate(component):
    ''' Function to derive the best and worst fold for a given dimension '''

    df = pd.read_csv(f'/home/nick/github_repos/Pollution-Autoencoders/data/test_metrics/pca/{component}_test_metrics.csv')
    # Outputs
    best_metrics = f'/home/nick/github_repos/Pollution-Autoencoders/data/test_metrics/pca/derived/{component}_best_metrics.csv'
    worst_metrics = f'/home/nick/github_repos/Pollution-Autoencoders/data/test_metrics/pca/derived/{component}_worst_metrics.csv'
    # Lists to write to file
    best_list=worst_list=['dim', 'variance', 'r2']

    # Write headers for best and worst metrics
    with open(best_metrics,'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(best_list)
        f.close() 
    with open(worst_metrics,'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(worst_list)
        f.close() 

    for i in range(189):
        # Reset best and worst variance and index for every dim
        best_var, worst_var = 0, 100
        best_idx=worst_idx=0

        # Retrieve a dict of each fold's variance for a given dimension
        fold_var = { i : df['variance'][i], i+189 : df['variance'][i+189], i+378 : df['variance'][i+378], 
            i+567 : df['variance'][i+567], i+756 : df['variance'][i+756] }
        
        # Search through the values for each given dim
        for key, val in fold_var.items():
            if val > best_var:
                best_var=val
                best_idx=key
            if val < worst_var:
                worst_var=val
                worst_idx=key

        # Save dim, variance, and r2 scores
        best_list = [i+2, df['variance'][best_idx], df['r2'][best_idx]]
        worst_list = [i+2, df['variance'][worst_idx], df['r2'][worst_idx]]
        print(f'Best fold for dim {i+2} has a score of {best_var}')
        print(f'Worst fold for dim {i+2} has a score of {worst_var}\n') 

        # Write values
        with open(best_metrics,'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(best_list)
            f.close() 
        with open(worst_metrics,'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(worst_list)
            f.close() 
            

def pca_run(dim, X, X_train, Y_train, component, cities):
    ''' 
    Run PCA using k-fold cross validation strategy

    @params:
        dim: Dimension to embed
        X: Normalized data to encode
        X_train, Y_train: Train data used in model creation
        component: Gas/particulate 
        cities: List of cities to append to the embedding
    '''

    vec_file = f'/home/nick/github_repos/Pollution-Autoencoders/data/vec/pca/{component}_vec.csv'

    # Train pca model
    model, encoded_train_data = pca(
        i=dim, 
        X_train=X_train, 
        Y_train=Y_train, 
        component=component)
    
    # Create full embedding
    encoded_data = model.transform(X)

    # Add city labels and save encoded data
    vec_labels = [f'dim_{i}' for i in range(1, dim+1)]
    vector_data = pd.DataFrame(data=encoded_data, columns=vec_labels)
    vector_data.insert(0, 'city', cities)
    vector_data.to_csv(path_or_buf=vec_file, index=None)


def main():
    ''' Set up soures and constants, call functions as needed '''
    
    ### Constants ###
    #component_names = ['co', 'no', 'no2', 'o3', 'so2', 'pm2_5', 'pm10', 'nh3']
    component_test = 'co'
    # Starting dimensions
    dims = 50
    # K-fold folds
    folds = 5

    ### Input files ###
    # Open normalized data and dependent, non-normalized data
    dfx = pd.read_csv(f"{os.environ['HOME']}/github_repos/Pollution-Autoencoders/data/data_norm/co_data_norm.csv")
    dfy = pd.read_csv(f"{os.environ['HOME']}/github_repos/Pollution-Autoencoders/data/data_clean/co_data_clean.csv")
    # City names to append
    cities = dfy['city'].values
    
    # Set x as the normalized values, y (non-normalized) as the daily average of final day
    X = dfx.values
    Y = dfy.loc[:, ['co_2021_06_06']].values
    # Split into train/test data
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=40)

    ### Function calls ###

    #pca_train_test(dims, X, Y, folds, component_test)
    pca_run(dims, X, X_train, Y_train, component_test, cities)
    #linreg.regression(X_train, X_test, Y_train, Y_test)


if __name__ == '__main__':
    main()