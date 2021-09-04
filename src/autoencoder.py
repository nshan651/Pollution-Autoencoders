import numpy as np
import pandas as pd
import itertools
import csv
import os
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import Normalizer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import r2_score


def autoencoder(dfx, y, dim, component, activation, lr=0.0001, batch=128, epochs=50):
    '''
    Autoencoder model
    
    @params:
        dfx: DataFrame of normalized x values
        y: Dependent variable to be predicted
        dim: Current dimension
        component: Name of gas or particulate
        activation: Activation function represented as a tuple pair
        lr: Learning rate, default of 0.0001
        batch: Batch size, default of 128
        epochs: Number of epochs, default of 50
    @return:
        variance_list: List of explainable variance scores
        r2_list: List of R squared scores
        encoded_data: The encoded X training data
    '''

    # Define the linear regression model
    regr = LinearRegression()

    print('---------- Autoencoder dim {} for {} ----------'.format(dim, component))
    # Training and test splits
    train, dev, train_labels, dev_labels = train_test_split(dfx, y, test_size=0.20, random_state=40)
    train, test, train_labels, test_labels = train_test_split(train, train_labels, test_size=0.20, random_state=40)
    input_data = Input(shape=(191,))

    x_train = train.loc[:, train.columns]
    x_test = test.loc[:, test.columns]
    x_dev = dev.loc[:, dev.columns]

    encoded = Dense(dim, activation=activation[0], name='bottleneck')(input_data)
    decoded = Dense(191, activation=activation[1])(encoded)

    autoencoder = Model(input_data, decoded)

    # Compile the autoencoder
    autoencoder.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=lr))
    autoencoder.summary()
    
    trained_model = autoencoder.fit(
        x_train, 
        x_train, 
        batch_size=batch, 
        epochs=epochs, 
        verbose=1,
        validation_data=(x_dev, x_dev)
    )
    
    # Bottleneck representation
    # Separate encoder that maps input to its encoded representation
    encoder = Model(autoencoder.input, autoencoder.get_layer('bottleneck').output)
    encoded_data = encoder.predict(x_train)    
    decoded_output = autoencoder.predict(x_train)  
    
    # Variance score explanation
    regr.fit(encoded_data, train_labels)

    encoded_data_test = encoder.predict(x_test)
    y_pred = regr.predict(encoded_data_test) # << Pass x_test to get predicitons for original uncompressed
    
    # Variance and r2 scores for the regression model
    variance = regr.score(encoded_data_test, test_labels) # << pass in x test for uncompressed values
    r2 = r2_score(test_labels, y_pred)
    print ('Variance score: %.2f' % variance)
    print ('R Square', r2)    

    return (variance, r2, encoded_data)
    

def ae_train(dims, component):
    '''
    Train the autoencoder model using cross-validation and grid search 
    
    @params:
        dims: Number of dimensions
        component_names: Gas/particulate list
    '''

    print('---------- Beginning Autoencoder training for {} ----------'.format(component))
    # Open normalized data
    dfx = pd.read_csv(f"{os.environ['HOME']}/github_repos/Pollution-Autoencoders/data/data_norm/{component}_data_norm.csv")
    # y value list using last day of 7-month data
    dfy = pd.read_csv(f"{os.environ['HOME']}/github_repos/Pollution-Autoencoders/data/data_clean/{component}_data_clean.csv")
    # Set x as the normalized values, y as the daily average of final day
    x = dfx.values
    y = dfy.loc[:, ['{}_2021_06_06'.format(component)]].values
    
    # Train Autoencoder model
    variance_list = []
    r2_list = []
    num_of_comp = list(range(2,dims+1))
    for i in num_of_comp:
        print('---------- Autoencoder dim {} for {} ----------'.format(i, component))
        variance, r2, X_train = autoencoder(
            dfx=dfx, 
            y=y, 
            dim=i, 
            component=component, 
            activation=('tanh', 'tanh'),
            lr=0.001,  
            batch=128,
            epochs=100
        )
        # Add r2/variance to lists
        variance_list.append(variance)
        r2_list.append(r2)


def ae_run(dims, component):
    '''
    Run the autoencoder model by performing a cross-validated linear
    regression on the encoded data
    
    @params:
        dims: Number of dimensions
        component_names: Gas/particulate list
    '''
    
    # Open normalized data
    dfx = pd.read_csv(f"{os.environ['HOME']}/github_repos/Pollution-Autoencoders/data/data_norm/{component}_data_norm.csv")
    # y value list using last day of 7-month data
    dfy = pd.read_csv(f"{os.environ['HOME']}/github_repos/Pollution-Autoencoders/data/data_clean/{component}_data_clean.csv")
    # City names to append
    cities = dfy['city'].values
    # Grid params to create model with
    param_grid = pd.read_csv(f'/home/nicks/github_repos/Pollution-Autoencoders/data/grid_params/hyperparams.csv')
    
    # Write headers for model metrics and vecs; erase previous model
    metrics_file = f'/home/nicks/github_repos/Pollution-Autoencoders/data/model_metrics/{component}_metrics.csv'
    vec_file = f'/home/nicks/github_repos/Pollution-Autoencoders/data/vec/{component}_vec.csv'
    with open(metrics_file,'w', newline='') as f:
        writer = csv.writer(f)
        metrics_labels = ['dim', 'variance', 'r2']
        writer.writerow(metrics_labels)
        f.close()  
    with open(vec_file,'w', newline='') as f:
        writer = csv.writer(f)
        dim_labels = [f'dim_{i}' for i in range(1, dims+1)]
        dim_labels.insert(0, 'city')
        writer.writerow(dim_labels)
        f.close()

    
    # Set x as the normalized values, y as the daily average of final day
    x = dfx.values
    y = dfy.loc[:, ['{}_2021_06_06'.format(component)]].values
    splits = 5
    # Number of features to compare
    num_of_dims=list(range(1,dims+1))
    
    # Define hyperparams and counter
    # Change to next set of hyperparams every 10 dims
    lr = param_grid['lr'][0]
    batch = param_grid['batch'][0]
    epochs = param_grid['epochs'][0]
    counter=0
    
    vector = []
    var_list = []
    r2_list = []

    for dim in num_of_dims:
        # Change to next set of hyperparams every 10 dims
        print(f'lr: {lr} batch: {batch} epochs: {epochs}')
        if dim%10==0 and dim<120:
            counter+=1
            lr = param_grid['lr'][counter]
            batch = param_grid['batch'][counter]
            epochs = param_grid['epochs'][counter]
            
        # Define the linear regression model
        regr = LinearRegression()
        print('---------- Autoencoder dim {} for {} ----------'.format(dim, component))
        
        input_data = Input(shape=(191,))

        # Create dense AE layers
        encoded = Dense(dim, activation='tanh', name='bottleneck')(input_data)
        decoded = Dense(191, activation='tanh')(encoded)

        autoencoder = Model(input_data, decoded)

        # Compile the autoencoder
        autoencoder.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=lr))
        autoencoder.summary()
        
        # Bottleneck representation
        # Separate encoder that maps input to its encoded representation
        encoder = Model(autoencoder.input, autoencoder.get_layer('bottleneck').output)
        encoded_data = encoder.predict(x)    
        decoded_output = autoencoder.predict(x)  
        
        # Variance score explanation
        regr.fit(encoded_data, y)

        encoded_data_test = encoder.predict(x)
        y_pred = regr.predict(encoded_data_test) 
        
        # Variance and r2 scores for the regression model
        variance = regr.score(encoded_data_test, y) 
        r2 = r2_score(y, y_pred)
        var_list.append(variance)
        r2_list.append(r2)
        vector = list(encoded_data)
        print (f'Variance score: {variance} for dim {dim}')
        print (f'R Square {r2} for dim {dim}')
   
    # Write all vector data 
    file_name = f'/home/nicks/github_repos/Pollution-Autoencoders/data/vec/{component}_vec.csv'
    vec_labels = [f'dim_{i}' for i in range(1, dims+1)]
    vector_data = pd.DataFrame(data=vector, columns=vec_labels)
    # Add city labels
    vector_data.insert(0, 'city', cities)
    vector_data.to_csv(path_or_buf=file_name, index=None)
    
    # Write variance and r2 scores of each gas for every dimension 
    output_dict = {'dim': num_of_dims, 'variance' : var_list, 'r2' : r2_list}
    file_name = f'/home/nicks/github_repos/Pollution-Autoencoders/data/model_metrics/{component}_metrics.csv'
    metrics_data = pd.DataFrame(data=output_dict)
    metrics_data.to_csv(path_or_buf=file_name, index=False)
    

def grid_search(x, y, splits, component, iter_dims, param_vec):
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
    # File name
    file_name = f'/home/nicks/github_repos/Pollution-Autoencoders/data/grid_params/{component}_vec_dim3'
    # Train/test and metrics for current component's set of data
    train_test = {}
    metrics={}
    # Contains all train_test splits/metrics in REG for n sets of splits
    train_test_dict = {}
    metrics_dict = {}
    metrics_dict_set = {}
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
        folds+=1
        # Update dict with train/test values
        train_test['X_train'] = X_train
        train_test['X_test'] = X_test
        train_test['Y_train'] = Y_train
        train_test['Y_test'] = Y_test
        # Save best sets of train/test data that have high R2 scores
        train_test_dict[folds] = train_test.copy()
        # Perform a grid search
        for dim in iter_dims:
            best_var = -100
            for vec in param_vec:
                variance, r2, _ = autoencoder(
                    dfx=dfx, 
                    y=y, 
                    dim=dim, 
                    component=component, 
                    activation=('tanh', 'tanh'),
                    lr=vec[0], 
                    batch=vec[1],
                    epochs=vec[2]
                )
                # Update with current highest variance
                if variance > best_var:
                    best_var = variance
                    grid_list = [folds, dim, variance, r2, vec[0], vec[1], vec[2]]
                
            print('--------------------')
            print(f'Best params for fold {grid_list[0]} dim {grid_list[1]}: variance={grid_list[2]} lr={grid_list[4]} batch={grid_list[5]} epochs={grid_list[6]}')        
            print('--------------------')
            # Update grid params files
            with open(file_name,'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(grid_list)
                f.close()         
        
    
### RUN ###

#COMPONENT_NAMES = ['co', 'no', 'no2', 'o3', 'so2', 'pm2_5', 'pm10', 'nh3']
COMPONENT_NAMES = ['co']
# Starting dimensions
DIMS = 190
# Grid search params
LR = [0.0001, 0.001, 0.01, 0.1]
BATCH = [32, 64, 128, 256]
EPOCH = [10, 50, 75, 100, 150]

# Param vector
PARAM_VEC = list(itertools.product(LR,BATCH,EPOCH))
#PARAM_VEC = list(itertools.product([0.01], [32], [10])) # test param vec
# List of key dimensions
#ITER_DIMS = np.array([2,3,4,5,10,15,20,25,30,40,50,60,80,100,120])
ITER_DIMS = np.arange(10, 121, 10)

### Function tests ###

## Grid Search ##

# Open normalized data
dfx = pd.read_csv(f"{os.environ['HOME']}/github_repos/Pollution-Autoencoders/data/data_norm/co_data_norm.csv")
# y value list using last day of 7-month data
dfy = pd.read_csv(f"{os.environ['HOME']}/github_repos/Pollution-Autoencoders/data/data_clean/co_data_clean.csv")
# Set x as the normalized values, y as the daily average of final day
x = dfx.values
y = dfy.loc[:, ['co_2021_06_06']].values
SPLITS = 5
#grid_search(x, y, SPLITS, 'co', ITER_DIMS, PARAM_VEC)

#ae_train(DIMS, COMPONENT_NAMES)
ae_run(DIMS, 'co') 
