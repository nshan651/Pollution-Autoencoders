import numpy as np
import pandas as pd
import keras
import tensorflow
import matplotlib.pyplot as plt

from csv import writer
from sklearn.ensemble import RandomForestRegressor
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, BatchNormalization
from tensorflow.keras import optimizers
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import explained_variance_score
from sklearn.preprocessing import Normalizer
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


def autoencoder(dfx, y, dims, component):

    regr = LinearRegression()
    variance_list = []
    r2_list = []
    for i in range(1, dims):
        print('---------- Autoencoder dim {} ----------'.format(i))
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

    # Write entry to file
    file_name = 'C:\\github_repos\\Universal-Embeddings\\data\\model_results\\autoencoders\\{}_ae_test.csv'.format(component)
    write_data = pd.DataFrame(data={'{}_var'.format(component): variance_list, '{}_r2'.format(component): r2_list})
    write_data.to_csv(path_or_buf=file_name, index=False)

    return (variance_list, r2_list)


def ae_linegraph(num_of_comp, component_names, colors_list):
    ''' 
    Plot a linegraph for pca
    '''
    
    plt.figure(figsize=(12,10))
    plt.rcParams.update({'font.size': 18})
    for i, component in enumerate(component_names):
        print('--- Autoencoder for {} ---'.format(component))
        # Open model folder
        file_name = 'C:\\github_repos\\Universal-Embeddings\\data\\model_results\\autoencoders\\{}_ae_test.csv'.format(component)
        model = pd.read_csv(filepath_or_buffer=file_name)
        #print(model['{}_var'.format(component)])
        #print(model['{}_r2'.format(component)])

        variance = model['{}_var'.format(component)]
        r2 = model['{}_r2'.format(component)]
        
        #variance, r2 = autoencoder(dfx, y, dims, component)
        plt.plot(num_of_comp, variance[:11], label = '{}'.format(component), linestyle = '-', marker = '+', color = colors_list[i])
        plt.plot(num_of_comp, r2[:11], linestyle = '-.', marker = 'H', color = colors_list[i])
        plt.xlabel('Dimension')
        plt.ylabel('Variance/R2')
        plt.title('Autoencoder of Polluting Gases')
        plt.legend()
    plt.show()

### RUN ###

COMPONENT_NAMES = ['co', 'no', 'no2', 'o3', 'so2', 'pm2_5', 'pm10', 'nh3']
#COMPONENT_NAMES = ['co']
COLORS_LIST = ['tab:blue', 'tab:green', 'tab:orange', 'tab:red', 'tab:purple', 'tab:cyan', 'tab:olive', 'tab:pink']
# Starting dimensions
DIMS = 190
# Number of components
#NUM_OF_COMP=list(range(2,191))
NUM_OF_COMP=list(range(2,191))

#ae_linegraph(NUM_OF_COMP, COMPONENT_NAMES, COLORS_LIST)

for i, component in enumerate(COMPONENT_NAMES):

    df = pd.read_csv('C:\\github_repos\\Universal-Embeddings\\data\\gases\\{}.csv'.format(component))
    columns = list(df.columns.values)

    # Features list and removal of city, lat, lon
    features = list(df.columns.values)
    del features[:1]
    del features[-1]

    # y value list using last day of 7-month data
    y = df.loc[:, ['{}_2021_06_06'.format(component)]].values

    # Normalize x values; save in data frame
    x = df.loc[:, features].values
    x = Normalizer().fit_transform(x)
    dfx = pd.DataFrame(x)

    print('--- Autoencoder for {} ---'.format(component))
    variance, r2 = autoencoder(dfx, y, DIMS, component)



