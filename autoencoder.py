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

    return (variance_list, r2_list, encoded_data)


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
        plt.plot(num_of_comp, variance, label = '{}'.format(component), linestyle = '-', marker = '+', color = colors_list[i])
        plt.plot(num_of_comp, r2, linestyle = '-.', marker = 'H', color = colors_list[i])
        plt.xlabel('Dimension')
        plt.ylabel('Variance/R2')
        plt.title('Autoencoder of Polluting Gases')
        
    plt.show()


def ae_scatter(component, colors_list):
    '''
    PCA scatter plot of multiple components 
    '''
    
    # Read in city data frame and select list of cities
    city_df = pd.read_csv(filepath_or_buffer='C:\\github_repos\\Universal-Embeddings\\data\\data_clean\\co_data_clean.csv')
    #annotations = city_df.loc[:,'city']
    annotations = pd.DataFrame(data=city_df)
    
    # Read in X_train values for first two dimensions
    data = pd.read_csv(filepath_or_buffer='C:\\github_repos\\Universal-Embeddings\\data\\model_results\\autoencoders\\co_xtrain_dims.csv')
    X_train = pd.DataFrame(data=data)
    X = X_train['co_X_train_ax1']
    Y = X_train['co_X_train_ax2']
   
    # Read in whitelisted city data
    #wlist_data = pd.read_csv(filepath_or_buffer='C:\\github_repos\\Universal-Embeddings\\data\\outliers_whitelist.csv')
    wlist_data = pd.read_csv(filepath_or_buffer='C:\\github_repos\\Universal-Embeddings\\data\\top200.csv')
    wlist = pd.DataFrame(data=wlist_data[:20])
    
    # Plot figure
    plt.figure(figsize=(14,14))
    plt.rcParams.update({'font.size': 18})
    #plt.scatter(X_train[:,0], X_train[:,1], label='{}'.format(component), c=colors_list[0], alpha=0.1)
    plt.scatter(X, Y, label='{}'.format(component), c=colors_list[0], alpha=0.1)

    size = len(X_train)
    plt.title('Carbon Monoxide Autoencoder in First Two Dimensions')
    
    #print(wlist)
    #print(wlist.iloc[0][0])
    #print(annotations.iloc[4400][0])
    #print(wlist.size)
    size = len(X_train)
    '''
    # Annotate points
    for i in range(size):
        for j in range(len(wlist)):
            if annotations.iloc[i][0] == wlist.iloc[j][0]:
                print(annotations.iloc[i][0])
                plt.annotate(text=annotations.iloc[i][0], xy=(X[i],Y[i]))
    '''
    plt.show()
    

def ae_heatmap(component, colors_list):
    # Read in component data and whitelist city names
    data = pd.read_csv('C:\\github_repos\\Universal-Embeddings\\data\\data_clean\\{}_data_clean.csv'.format(component))
    df = pd.DataFrame(data=data)
    wlist_data = pd.read_csv(filepath_or_buffer='C:\\github_repos\\Universal-Embeddings\\data\\outliers_whitelist.csv')
    wlist = pd.DataFrame(data=wlist_data[:10])

    annotations = df['city']

    # Remove lat/lon
    df.drop(['lat', 'lon'], 1, inplace=True) 

    #print(df.head())
    ts_list = []
    val_dict = {}
    total = 0
    
    cities_list = []
    val_list = []
   # print(annotations.iloc[1])
    #print(wlist.iloc[1][0])

    # Loop through and find indexes of each city in data
    
    for i in range(len(annotations)):
        for w in range(len(wlist)):
            # Check if current city is in the whitelist
            if annotations.iloc[i] == wlist.iloc[w][0]:
                # Average for seven days
                for c in range(1, len(df.columns)):
                    total+=df.iloc[w][c]
                    if (c%7 == 0):   
                        weekly = round(total/7, 5)
                        ts_list.append(weekly)
                        total = 0
                # Get the values list and the
                val_list.append(ts_list)
                cities_list.append(annotations.iloc[i])
                # Clear list for next iteration
                ts_list = []
    #print(val_dict)
    
    # Plot figure
    plt.rcParams.update({'font.size': 18})
    fig, ax = plt.subplots(figsize=(20,18))
    #plt.figure(figsize=(14,14))
    #ax.plot(np.arange(5))
    heat = ax.imshow(val_list, cmap='inferno')
    #plt.imshow(val_dict)
    
    # Set axis width
    #ax.tick_params(axis='both', which='major', length=10, width=25)
    ax.set_xticks(np.arange(0, 27, 2)) # 27 weeks
    ax.set_yticks(np.arange(len(cities_list)))
    ax.set_yticklabels(cities_list)

    '''
    # Loop over data dimensions and create text annotations.
    for i in range(len(cities_list)):
        for j in range(27):
            text = ax.text(j, i, val_list[i][j],
                        ha="center", va="center", color="w")
    '''
    fig.colorbar(heat)
    plt.show()
    # Set x and y
    
    
    #plt.title('Carbon Monoxide Autoencoder in First Two Dimensions')

### RUN ###

#COMPONENT_NAMES = ['co', 'no', 'no2', 'o3', 'so2', 'pm2_5', 'pm10', 'nh3']
COMPONENT_NAMES = ['co']
COLORS_LIST = ['tab:blue', 'tab:green', 'tab:orange', 'tab:red', 'tab:purple', 'tab:cyan', 'tab:olive', 'tab:pink']
# Starting dimensions
DIMS = 3
# Number of components
#NUM_OF_COMP=list(range(2,191))
NUM_OF_COMP=list(range(2,191))

#ae_scatter(COMPONENT_NAMES, COLORS_LIST)
ae_heatmap('co', COLORS_LIST)
'''
for i, component in enumerate(COMPONENT_NAMES):

    df = pd.read_csv('C:\\github_repos\\Universal-Embeddings\\data\\data_clean\\{}_data_clean.csv'.format(component))
    columns = list(df.columns.values)

    # Features list and removal of city, lat, lon
    features = list(df.columns.values)
    del features[:3] # << Change for clean data
    del features[-1]

    # y value list using last day of 7-month data
    y = df.loc[:, ['{}_2021_06_06'.format(component)]].values

    # Normalize x values; save in data frame
    x = df.loc[:, features].values
    x = Normalizer().fit_transform(x)
    dfx = pd.DataFrame(x)

    print('--- Autoencoder for {} ---'.format(component))
    variance, r2, X_train = autoencoder(dfx, y, DIMS, component)
    #ae_scatter(X_train, component, COLORS_LIST)

    # Write entry to file
    file_name = 'C:\\github_repos\\Universal-Embeddings\\data\\model_results\\autoencoders\\{}_ae_test.csv'.format(component)
    write_data = pd.DataFrame(data={'{}_X_train_ax1'.format(component) : X_train[:,0], '{}_X_train_ax2'.format(component) : X_train[:,1]})
    write_data.to_csv(path_or_buf=file_name, index=False)
    #ae_scatter(X_train, COMPONENT_NAMES, COLORS_LIST)
'''
