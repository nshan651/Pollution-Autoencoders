import numpy as np
import pandas as pd
import itertools
import csv
import os
import linreg # src/linreg.py
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import Normalizer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import r2_score


def autoencoder(dim, X_train, Y_train, component, activation, lr, batch, epochs):
    '''
    Autoencoder model
    
    @params:
        dim: Current dimension
        X_train, Y_train: training data
        component: Name of gas or particulate
        activation: Activation function represented as a tuple pair
        lr: Learning rate
        batch: Batch size
        epochs: Number of epochs
    @return:
        encoded_layer: The encoded layer responsible for encoding the data
        encoded_train_data: Training embeddings needed to fit the regression model
        Y_train: The reduced Y training data to match size of the encoded training data
    '''

    # Additional split of train/test data; keep dev set for validation
    X_train, X_dev, Y_train, Y_dev = train_test_split(X_train, Y_train, test_size=0.20, random_state=40)
    input_data = Input(shape=(191,))

    # Create layers
    encoded = Dense(dim, activation=activation[0], name='bottleneck')(input_data)
    decoded = Dense(191, activation=activation[1])(encoded)

    # Define the autoencoder model
    autoencoder = Model(input_data, decoded)

    # Compile the autoencoder
    autoencoder.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=lr))
    autoencoder.summary()
    
    # Fit the model
    autoencoder.fit(
        X_train, 
        X_train, 
        batch_size=batch, 
        epochs=epochs, 
        verbose=1,
        validation_data=(X_dev, X_dev)  # Test on dev set, update weights
    )

    # Compare encode layer embeddings to training data
    encoded_layer = Model(autoencoder.input, autoencoder.get_layer('bottleneck').output)
    encoded_train_data = encoded_layer.predict(X_train)  
    
    return encoded_layer, encoded_train_data, Y_train


def ae_train_test(dims, X_train, X_test, Y_train, Y_test, component, param_grid):
    ''' 
    Train and test autoencoder model 
    
    @params:
        dims: Number of dimensions to train
        component: Gas/particulate
        param_grid: List of key hyperparameters to test 
        production: Boolean to determine whether to train/test or to create full embedding
    '''

    # Train Autoencoder model
    variance_list = []
    r2_list = []
    num_of_dims = list(range(2,dims+1))

    # Define hyperparams and counter
    lr = param_grid['lr'][0]
    batch = param_grid['batch'][0]
    epochs = param_grid['epochs'][0]
    # List of key dimensions to advance counter
    param_dims = list(param_grid['dim'])
    counter=0

    for dim in num_of_dims:
        print(f'dim {dim} || lr: {lr} || batch: {batch} || epochs: {epochs}')
        # Use key hyperparameters in the training process
        if dim in param_dims:
            lr = param_grid['lr'][counter]
            batch = param_grid['batch'][counter]
            epochs = param_grid['epochs'][counter]
            counter+=1
        # Train the model
        encoded_layer, encoded_train_data, Y_reduced_train = autoencoder(
            dim=dim,
            X_train=X_train,
            Y_train=Y_train,
            component=component,
            activation=('tanh', 'tanh'),
            lr=lr, 
            batch=batch,
            epochs=epochs
        )

        # Encode test data
        encoded_test_data = encoded_layer.predict(X_test)
        # Perform a linear regression based off encoded data
        variance, r2 = linreg.regression(encoded_train_data, encoded_test_data, Y_reduced_train, Y_test)

        # Save model metrics
        variance_list.append(variance)
        r2_list.append(r2)

    # Write variance and r2 scores of each gas for every dimension 
    output_dict = {'dim': num_of_dims, 'variance' : variance_list, 'r2' : r2_list}
    metrics_data = pd.DataFrame(data=output_dict)
    metrics_data.to_csv(path_or_buf=metrics_file, index=False)


def ae_run(dim, X, X_train, Y_train, lr, batch, epochs, component, cities):
    '''
    Run the autoencoder model by performing a cross-validated linear
    regression on the encoded data
     
    @params:
        dim: Dimension to embed
        component: Gas/particulate 
        cities: List of cities to append to the embedding
        param_grid: The correct hyperparameters to be used. Note the dimensions that were tested in the 
            grid search. If dimension was not tested for it, use hyperparams of last tested dimension
    '''

    # File to save model metrics to
    vec_file = f'/home/nick/github_repos/Pollution-Autoencoders/data/vec/{component}_vec_{dim}.csv'

    # Train the model that will be used to create the embedding based on desired input dimensions
    encoded_layer, *_ = autoencoder(
        dim=dim,
        X_train=X_train,
        Y_train=Y_train,
        component=component,
        activation=('tanh', 'tanh'),
        lr=lr, 
        batch=batch,
        epochs=epochs
    )
   
    # Create encoded data based off full dataset
    encoded_data = encoded_layer.predict(X)
    
    # Add city labels and save encoded data
    vec_labels = [f'dim_{i}' for i in range(1, dim+1)]
    vector_data = pd.DataFrame(data=encoded_data, columns=vec_labels)
    vector_data.insert(0, 'city', cities)
    vector_data.to_csv(path_or_buf=vec_file, index=None)


def grid_search(x, y, folds, component, iter_dims, key_params):
    '''
    Run the linear regression for a single gas
    using k-fold cross validation strategy

    @params:
        x: The normalized x values to be used in the embedding
        y: The dependent variable
        folds: Number of folds to test
        component: Name of gas or particulate
        activation: Activation function represented as a tuple pair
        lr: Learning rate
        batch: Batch size
        epochs: Number of epochs        
        iter_dims: Key dimensions to perform the grid search on
        key_params: list of key parameters to test
    '''

    print(f'---------- Beginning Linear Regression for {component} ----------')

    # k-fold cross validation
    kfold = KFold(n_splits=folds, shuffle=True, random_state=10)
    fold_count=0
    # File name
    file_name = f'/home/nick/github_repos/Pollution-Autoencoders/data/grid_params/{component}_vec.csv'
    
    # Headers for grid search
    grid_list = ['fold', 'dim', 'variance', 'r2', 'lr', 'batch', 'epochs']
    # Write header
    with open(file_name,'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(grid_list)
        f.close() 

    # Loop through train/test data and save the best data with highest R2 scores
    for training_index, test_index in kfold.split(x):
        # Split X and Y train and test data
        X_train, X_test = x[training_index, :], x[test_index, :]
        Y_train, Y_test = y[training_index], y[test_index]
        fold_count+=1

        # Perform a grid search
        for dim in iter_dims:
            best_var = -100
            for vec in key_params:
                # Train model
                encoded_layer, encoded_train_data, Y_reduced_train = autoencoder(
                    dim=dim,
                    X_train=X_train,
                    Y_train=Y_train,
                    component=component,
                    activation=('tanh', 'tanh'),
                    lr=lr, 
                    batch=batch,
                    epochs=epochs
                )

                # Encode test data
                encoded_test_data = encoded_layer.predict(X_test)
                # Perform a linear regression based off the encoded data
                variance, r2 = linreg.regression(encoded_train_data, encoded_test_data, Y_reduced_train, Y_test)
                
                # Update with current highest variance
                if variance > best_var:
                    best_var = variance
                    grid_list = [fold_count, dim, variance, r2, vec[0], vec[1], vec[2]]
                
            print(f'Best params for fold {grid_list[0]} dim {grid_list[1]}: variance={grid_list[2]} lr={grid_list[4]} batch={grid_list[5]} epochs={grid_list[6]}')        
            
            # Update grid params files
            with open(file_name,'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(grid_list)
                f.close()


def main():
    ''' Set up soures and constants, call functions as needed '''
    
    ### Constants ###
    #component_names = ['co', 'no', 'no2', 'o3', 'so2', 'pm2_5', 'pm10', 'nh3']
    component_test = 'co'
    # Starting dimensions
    dims = 50
    # K-fold folds
    folds = 5
    # Grid search params
    lr = [0.0001, 0.001, 0.01, 0.1]
    batch = [32, 64, 128, 256]
    epochs = [10, 50, 75, 100, 150]
    # Param vector
    key_params = list(itertools.product(lr,batch,epochs))
    # List of key dimensions to perform grid search on
    iter_dims = np.arange(1, 11, 1)

    ### Input files ###
    # Open normalized data and dependent, non-normalized data
    dfx = pd.read_csv(f"{os.environ['HOME']}/github_repos/Pollution-Autoencoders/data/data_norm/co_data_norm.csv")
    dfy = pd.read_csv(f"{os.environ['HOME']}/github_repos/Pollution-Autoencoders/data/data_clean/co_data_clean.csv")
    # City names to append
    cities = dfy['city'].values
    # Grid params to create model with
    param_grid = pd.read_csv(f'/home/nick/github_repos/Pollution-Autoencoders/data/grid_params/hyperparams.csv')
    
    # Set x as the normalized values, y (non-normalized) as the daily average of final day
    X = dfx.values
    Y = dfy.loc[:, ['co_2021_06_06']].values
    # Split into train/test data
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=40)

    ### Function calls ###
    
    #grid_search(X, Y, folds, 'co', iter_dims, key_params)
    #ae_train_test(dims, X_train, X_test, Y_train, Y_test, component_test, param_grid)
    #ae_run(dims, X, X_train, Y_train, 0.01, 64, 75, component_test, cities)
    #linreg.regression(X_train, X_test, Y_train, Y_test)


if __name__ == '__main__':
    main()