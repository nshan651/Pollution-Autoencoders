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
        #columns = list(df.columns.values)

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
        print('--- Autoencoder for {} ---'.format(component))
        variance, r2, X_train = autoencoder(dfx, y, dims, component)

        # Create empty list of NaNs
        append_size = len(X_train[:,0]) - dims + 1
        norm = np.empty(append_size)
        norm[:] = np.NaN
        
        # Normalize var and r2 so they are same length as ax1 and ax2
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
    


def ae_linegraph(dims, component_names, colors_list):
    ''' 
    Plot the explained variance across dimensions using the model results

    @params:
        dims: Number of dimensions
        component_names: Gas/particulate list
        colors_list: List of colors to be used
    '''
    
    num_of_comp = list(range(2,dims+1))
    plt.figure(figsize=(12,10))
    plt.rcParams.update({'font.size': 18})
    for i, component in enumerate(component_names):
        print('--- Autoencoder for {} ---'.format(component))
        # Open model folder
        file_name = '/home/nicks/github_repos/Pollution-Autoencoders/data/model_results/autoencoders/{}_ae_results.csv'.format(component)
        model = pd.read_csv(filepath_or_buffer=file_name)
        #print(model['{}_var'.format(component)])
        #print(model['{}_r2'.format(component)])

        variance = model['{}_var'.format(component)]
        r2 = model['{}_r2'.format(component)]
        
        #variance, r2 = autoencoder(dfx, y, dims, component)
        plt.plot(num_of_comp, variance[:dims-1], label = '{}'.format(component), linestyle = '-', marker = '+', color = colors_list[i])
        plt.plot(num_of_comp, r2[:dims-1], linestyle = '-.', marker = 'H', color = colors_list[i])
        plt.xlabel('Dimension')
        plt.ylabel('% Explained Variance')
        plt.title('Autoencoder of Polluting Gases')
        
    plt.show()


def ae_scatter(component, color):
    '''
    Autoencoder scatter plot of multiple components for the first two dimensions

    @params:
        component: Name of gas or particulate
        color: Scatter plot color
    '''

    # Read in X_train values for first two dimensions from model reuslts
    data = pd.read_csv(filepath_or_buffer='/home/nicks/github_repos/Pollution-Autoencoders/data/model_results/autoencoders/{}_ae_results.csv'.format(component))
    X_train = pd.DataFrame(data=data)
    X = X_train['{}_X_train_ax1'.format(component)]
    Y = X_train['{}_X_train_ax2'.format(component)]

    # Read in city data frame and select list of cities
    city_df = pd.read_csv(filepath_or_buffer='/home/nicks/github_repos/Pollution-Autoencoders/data/data_clean/{}_data_clean.csv'.format(component))
    annotations = pd.DataFrame(data=city_df)
   
    # Read in whitelisted city data
    # Whitelist can be top200 cities or the outliers
    wlist_data = pd.read_csv(filepath_or_buffer='/home/nicks/github_repos/Pollution-Autoencoders/data/other/top200.csv')
    wlist = pd.DataFrame(data=wlist_data[:20])
    
    # Plot figure
    plt.figure(figsize=(14,14))
    plt.rcParams.update({'font.size': 18})
    plt.scatter(X, Y, label='{}'.format(component), c=color, alpha=0.1)

    size = len(X_train)
    plt.title('Carbon Monoxide Autoencoder in First Two Dimensions')
    '''
    # Annotate points (ad hoc)
    for i in range(size):
        for j in range(len(wlist)):
            if annotations.iloc[i][0] == wlist.iloc[j][0]:
                print(annotations.iloc[i][0])
                plt.annotate(text=annotations.iloc[i][0], xy=(X[i],Y[i]))
    '''
    plt.show()
    

def ae_heatmap(component, colors_list):
    # Read in component data and create an annotation list of cities
    data = pd.read_csv('/home/nicks/github_repos/Pollution-Autoencoders/data/data_clean/{}_data_clean.csv'.format(component))
    df = pd.DataFrame(data=data)
    annotations = df['city']

    # Read in whitelist of cities to graph
    wlist_data = pd.read_csv(filepath_or_buffer='/home/nicks/github_repos/Pollution-Autoencoders/data/other/outliers_whitelist.csv')
    wlist = pd.DataFrame(data=wlist_data[:10])
    
    # Remove lat/lon
    df.drop(['lat', 'lon'], 1, inplace=True) 

    # Time series data, city labels, and values list
    ts_list = []
    cities_list = []
    val_list = []
    total = 0
    
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
                # Update value and city name
                val_list.append(ts_list)
                cities_list.append(annotations.iloc[i])
                # Clear list for next iteration
                ts_list = []
    
    # Plot figure
    plt.rcParams.update({'font.size': 18})
    fig, ax = plt.subplots(figsize=(15,14))
    heat = ax.imshow(val_list, cmap='inferno')
    
    # Set axis width
    ax.set_xticks(np.arange(0, 27, 2)) # 27 weeks
    ax.set_yticks(np.arange(len(cities_list)))
    ax.set_yticklabels(cities_list)
    
    fig.colorbar(heat)
    plt.show()
    

### RUN ###

COMPONENT_NAMES = ['co', 'no', 'no2', 'o3', 'so2', 'pm2_5', 'pm10', 'nh3']
#COMPONENT_NAMES = ['no2']
COLORS_LIST = ['tab:blue', 'tab:green', 'tab:orange', 'tab:red', 'tab:purple', 'tab:cyan', 'tab:olive', 'tab:pink']
# Starting dimensions; Change this to edit
DIMS = 190

# Method tests
#ae_run(DIMS, ['co'])
ae_linegraph(DIMS, COMPONENT_NAMES, COLORS_LIST)
ae_scatter(COMPONENT_NAMES[0], COLORS_LIST[0])
ae_heatmap(COMPONENT_NAMES[0], COLORS_LIST)

