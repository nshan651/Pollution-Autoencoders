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


def autoencoder(dfx, y, dims, component):
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

        x_train = train.loc[:, train.columns]
        x_test = test.loc[:, test.columns]
        x_dev = dev.loc[:, dev.columns]

        encoded = Dense(i, activation='tanh', name='bottleneck')(input_data)
        decoded = Dense(191, activation='tanh')(encoded)

        autoencoder = Model(input_data, decoded)

        # Compile the autoencoder
        autoencoder.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=0.0001))
        autoencoder.summary()

        trained_model = autoencoder.fit(
            x_train, 
            x_train, 
            batch_size=128, 
            epochs=50, 
            verbose=1,
            validation_data=(x_dev, x_dev)
        )

        # Bottleneck representation
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
        r2_val = r2_score(test_labels, y_pred)
        print ('Variance score: %.2f' % variance)
        print ('R Square', r2_val)

        # Add r2/variance to lists
        variance_list.append(variance)
        r2_list.append(r2_val)        

    return (variance_list, r2_list, encoded_data)


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
        del features[-1]

        # y value list using last day of 7-month data
        y = df.loc[:, ['{}_2021_06_06'.format(component)]].values

        # Normalize x values; save in data frame
        x = df.loc[:, features].values
        x = Normalizer().fit_transform(x)
        dfx = pd.DataFrame(x)

        # Train Autoencoder model
        variance, r2, X_train = autoencoder(dfx, y, dims, component)

        # Normalize var and r2 so they are same length as ax1 and ax2
        append_size = len(X_train[:,0]) - dims + 1
        norm = np.empty(append_size)
        norm[:] = np.NaN
        var_norm = [*variance, *norm]
        r2_norm = [*r2, *norm]

        # Output dict to write 
        output_dict = {
            '{}_X_train_ax1'.format(component): X_train[:,0],
            '{}_X_train_ax2'.format(component) : X_train[:,1],
            '{}_var'.format(component) : var_norm,
            '{}_r2'.format(component) : r2_norm
        }

        # Write entry to file
        file_name = '/home/nicks/github_repos/Pollution-Autoencoders/data/model_results/autoencoders/{}_ae_results.csv'.format(component)
        write_data = pd.DataFrame(data=output_dict)
        write_data.to_csv(path_or_buf=file_name, index=False)
    
### RUN ###

#COMPONENT_NAMES = ['co', 'no', 'no2', 'o3', 'so2', 'pm2_5', 'pm10', 'nh3']
COMPONENT_NAMES = ['no2']
COLORS_LIST = ['tab:blue', 'tab:green', 'tab:orange', 'tab:red', 'tab:purple', 'tab:cyan', 'tab:olive', 'tab:pink']
# Starting dimensions; Change this to edit
DIMS = 3

# Method tests
ae_run(DIMS, COMPONENT_NAMES)


