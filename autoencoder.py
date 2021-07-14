import numpy as np
import pandas as pd
import itertools
import matplotlib.pyplot as plt
import csv

from numpy.core.numeric import NaN
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import Normalizer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
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
    input_data = Input(shape=(121,))

    x_train = train.loc[:, train.columns]
    x_test = test.loc[:, test.columns]
    x_dev = dev.loc[:, dev.columns]

    encoded = Dense(dim, activation=activation[0], name='bottleneck')(input_data)
    decoded = Dense(121, activation=activation[1])(encoded)

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
    decoded_output_test = autoencoder.predict(x_test)
    
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
    

def ae_run(dims, component_names):
    '''
    Run and train the autoencoder model
    
    @params:
        dims: Number of dimensions
        component_names: Gas/particulate list
    '''

    for component in component_names:
        print('---------- Beginning Autoencoder training for {} ----------'.format(component))
        df = pd.read_csv('/home/nicks/github_repos/Pollution-Autoencoders/data/data_clean/{}_data_clean.csv'.format(component))
        
        # Features list and removal of city, lat, lon
        features = list(df.columns.values)
        del features[:3] 
        #del features[-1]
        del features[121:]

        # y value list using last day of 7-month data
        y = df.loc[:, ['{}_2021_03_28'.format(component)]].values
        print(features)
        
        # Normalize x values; save in data frame
        x = df.loc[:, features].values
        x = Normalizer().fit_transform(x)
        dfx = pd.DataFrame(x)

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
                lr=0.0001,  # <<< Temp values
                batch=128,
                epochs=100
            )
            # Add r2/variance to lists
            variance_list.append(variance)
            r2_list.append(r2)  
        # Normalize var and r2 so they are same length as ax1 and ax2
        append_size = len(X_train[:,0]) - dims + 1
        norm = np.empty(append_size)
        norm[:] = np.NaN
        var_norm = [*variance_list, *norm]
        r2_norm = [*r2_list, *norm]

        # Output dict to write 
        output_dict = {
            '{}_X_train_ax1'.format(component): X_train[:,0],
            '{}_X_train_ax2'.format(component) : X_train[:,1],
            '{}_var'.format(component) : var_norm,
            '{}_r2'.format(component) : r2_norm
        }

        # Write entry to file
        file_name = '/home/nicks/Documents/model_tuning_results/autoencoders/test/{}_ae_results.csv'.format(component)
        write_data = pd.DataFrame(data=output_dict)
        write_data.to_csv(path_or_buf=file_name, index=False)


def grid_search(component, iter_dims, param_vec):
    
    print('---------- Beginning Autoencoder training for {} ----------'.format(component))
    df = pd.read_csv('/home/nicks/github_repos/Pollution-Autoencoders/data/data_clean/{}_data_clean.csv'.format(component))
    
    # Features list and removal of city, lat, lon
    features = list(df.columns.values)
    del features[:3] 
    #del features[-1]
    del features[121:]

    # y value list using last day of 7-month data
    y = df.loc[:, ['{}_2021_03_28'.format(component)]].values
    
    # Normalize x values; save in data frame
    x = df.loc[:, features].values
    x = Normalizer().fit_transform(x)
    dfx = pd.DataFrame(x)
    
    # List of best params for a given dim
    grid_list = ['dim', 'variance', 'r2', 'lr', 'batch', 'epochs'] 
    # Write title and erase any previous values
    # Create title
    file_name = '/home/nicks/Documents/model_tuning_results/autoencoders/vector_dimensions/{}_vec_dim'.format(component)
    #with open(file_name,'w') as f:
    #    writer = csv.writer(f)
    #    writer.writerow(grid_list)
    # Perform a Grid Search (3875 cycles)
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
                grid_list = [dim, variance, r2, vec[0], vec[1], vec[2]]
        # Update vector dimension files
        with open(file_name,'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(grid_list)    
        print('Best for dim {}: variance={} lr={} batch={} epochs={}'.format(
            grid_list[0], grid_list[1], grid_list[3], grid_list[4], grid_list[5]))
        
    
### RUN ###

#COMPONENT_NAMES = ['co', 'no', 'no2', 'o3', 'so2', 'pm2_5', 'pm10', 'nh3']
COMPONENT_NAMES = ['co']
COLORS_LIST = ['tab:blue', 'tab:green', 'tab:orange', 'tab:red', 'tab:purple', 'tab:cyan', 'tab:olive', 'tab:pink']
# Starting dimensions; Change this to edit
DIMS = 3
# Grid search params
LR = [0.0001, 0.001, 0.01, 0.1]
BATCH = [32, 64, 128, 256]
EPOCH = [10, 50, 75, 100]
# Param vector
PARAM_VEC = list(itertools.product(LR,BATCH,EPOCH))
# List of key dimensions
ITER_DIMS = np.concatenate((np.arange(14,26,1), np.array([30,40,50,60,80,100,120])), axis=0)

# Method tests
#ae_run(DIMS, COMPONENT_NAMES)
grid_search('co', ITER_DIMS, PARAM_VEC)

