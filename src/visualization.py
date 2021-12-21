import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import itertools


def linegraph(type, dims, component_names, colors_list):
    ''' 
    Plot the explained variance across dimensions using the model results

    @params:
        f_name: Name of the file
        type: Type of reduction method, pca or autoencoders
        dims: Number of dimensions
        component_names: Gas/particulate list
        colors_list: List of colors to be used
    '''
    
    num_of_comp = list(range(2,dims+1)) # TODO: delete this later
    num_of_dims = list(range(2, dims+1))
    # Plot results
    plt.figure(figsize=(12,10))
    for i, component in enumerate(component_names):
        # AE or PCA
        if type == 'ae':
            #file_name = f'{f_name}{component}_metrics.csv'
            file_name = f'/home/nick/github_repos/Pollution-Autoencoders/data/model_metrics/ae/{component}_metrics.csv'
            plt_title = 'Autoencoder Reduced Representation of Air Pollutants'
        elif type == 'pca':
            file_name = f'/home/nick/github_repos/Pollution-Autoencoders/data/model_metrics/pca/{component}_metrics'
            plt_title = 'PCA Reduced Representation of Air Pollutants'
        else:
            print('Type must be "ae" or "pca"')
            quit()

        # Read in model results
        model = pd.read_csv(filepath_or_buffer=file_name)
        variance = model['variance']
        r2 = model['r2']
        
        # sns
        ax = sns.lineplot(x=num_of_dims, y=variance[:dims-1], linewidth=1, label=f'{component}', color=colors_list[i])
        sns.lineplot(x=num_of_dims, y=r2[:dims-1], linewidth=5, linestyle = '-.', marker = 'H', label=f'{component}', color=colors_list[i])
        sns.despine()
        
        plt.rcParams.update({'font.size': 22})
        plt.tick_params(axis='both', which='major', labelsize=28)
        ax.set_xlabel('Dimension', fontsize='large')
        ax.set_ylabel('% Explained Variance', fontsize='large')
        plt.title(plt_title)

        #plt.plot(num_of_comp, variance[:dims-1], label = '{}'.format(component), linestyle = '-', marker = '+', color = colors_list[i])
        #plt.plot(num_of_comp, r2[:dims-1], linestyle = '-.', marker = 'H', color = colors_list[i])
        #plt.rcParams.update({'font.size': 22})
        #plt.tick_params(axis='both', which='major', labelsize=28)
        #plt.xlabel('Dimension')
        #plt.ylabel('% Explained Variance')
        #plt.xlabel('')
        #plt.ylabel('')
        #plt.title(plt_title)
        plt.ylim([0,1])
        plt.legend()
    plt.show()


def metrics_comparison(type, dims, component_names, colors_list):
    ''' 
    Plot the explained variance across dimensions using the model results

    @params:
        type: Type of reduction method, pca or autoencoders
        dims: Number of dimensions
        component_names: Gas/particulate list
        colors_list: List of colors to be used
    '''
     
    num_of_comp = list(range(2,dims+1)) # TODO: delete this later
    num_of_dims = list(range(1, dims+1))
    # Plot results
    plt.figure(figsize=(12,10))
    for i, component in enumerate(component_names):
        # AE or PCA
        if type == 'ae':
            high_file = f'/home/nick/github_repos/Pollution-Autoencoders/data/modle_metrics/ae/best_worst/{component}_best_metrics.csv'
            low_file = f'/home/nick/github_repos/Pollution-Autoencoders/data/model_metrics/ae/best_worst/{component}_worst_metrics.csv'
            plt_title = 'Autoencoder Reduced Representation of Air Pollutants'
        elif type == 'pca':
            high_file = f'/home/nick/github_repos/Pollution-Autoencoders/data/model_metrics/pca/best_worst/{component}_best_metrics.csv'
            low_file = f'/home/nick/github_repos/Pollution-Autoencoders/data/model_metrics/pca/best_worst/{component}_worst_metrics.csv'
            plt_title = 'PCA Reduced Representation of Air Pollutants'
        else:
            print('Type must be "ae" or "pca"')
            quit()

        # Read in model results
        best = pd.read_csv(filepath_or_buffer=high_file)
        worst = pd.read_csv(filepath_or_buffer=low_file)
        # Best and worst scores read in
        best_variance = best['variance']
        best_r2 = best['r2']
        worst_variance = worst['variance']
        worst_r2 = worst['r2']

        # Plots
        plt.plot(num_of_comp, best_variance[:dims-1], label = f'{component} high variance', linestyle = '-', marker = '+', color = 'green')
        plt.plot(num_of_comp, best_r2[:dims-1], linestyle = '-.', marker = 'H', color = 'green')
        plt.plot(num_of_comp, worst_variance[:dims-1], label = f'{component} low variance', linestyle = '-', marker = '+', color = 'red')
        plt.plot(num_of_comp, worst_r2[:dims-1], linestyle = '-.', marker = 'H', color = 'red')
        plt.rcParams.update({'font.size': 22})
        plt.tick_params(axis='both', which='major', labelsize=28)

        plt.xlabel('Dimension')
        plt.ylabel('% Explained Variance')
        plt.title('High-low variance scores')
        plt.ylim([0,1])
        plt.legend()
    plt.show()

def scatter(f_name, type, component, color):
    '''
    Autoencoder scatter plot of multiple components for the first two dimensions

    @params:
        f_name: Name of the file
        component: Name of gas or particulate
        color: Scatter plot color
    '''

    # Read in X_train values for first two dimensions from model results
    if type == 'ae':
        file_name = '%s/autoencoders/trail1/%s_ae_results_gs.csv' % (f_name, component)
    elif type == 'pca':
        file_name = '%s/autoencoders/trial1/%s_pca_results_gs.csv' % (f_name, component)
    else:
        print('Type must be "ae" or "pca"')
        quit()

    model = pd.read_csv(filepath_or_buffer=file_name)
    X_train = pd.DataFrame(data=model)
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


def heatmap(component):
    # Read in component data and create an annotation list of cities
    data = pd.read_csv('/home/nicks/github_repos/Pollution-Autoencoders/data/data_clean/{}_data_clean.csv'.format(component))
    df = pd.DataFrame(data=data)
    annotations = df['city']

    # Read in whitelist of cities to graph
    wlist_data = pd.read_csv(filepath_or_buffer='/home/nick/github_repos/Pollution-Autoencoders/data/other/outliers_whitelist.csv')
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
    heat = ax.imshow(val_list, cmap='plasma')
    
    # Set axis width
    ax.set_xticks(np.arange(0, 27, 2)) # 27 weeks
    ax.set_yticks(np.arange(len(cities_list)))
    ax.set_yticklabels(cities_list)
    
    fig.colorbar(heat)
    plt.show()

def correlation(X_data_file, Y_data_file, component):
    SIZE = 190
    
    # Read in normalized data
    norm_data = pd.read_csv(filepath_or_buffer=X_data_file)
   
    X_matrix, Y_matrix = [], []
    norm_labels = ['dim_{}'.format(i) for i in range(1, 190+1)]
    
    
    for i in range(SIZE):
        X_matrix.append(norm_data[norm_labels[i]][:])
    
    coefs = np.corrcoef(X_matrix)
    print(coefs.shape)

    mask = np.zeros_like(coefs)
    mask[np.triu_indices_from(mask)] = True
    plt.rcParams.update({'font.size': 18})
    plt.subplots(figsize=(15,12))
    plt.title(f'Correlation Matrix for {component}')
    # Seaborn style
    ax = sns.heatmap(
        coefs, 
        mask=mask,
        vmin=-1, vmax=1, center=0,
        cmap='coolwarm',
        xticklabels=15,
        yticklabels=15,
        square=True
    )
    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=45,
        horizontalalignment='right'
    )

    plt.show()
    
    ''' Matplot traditional graph
    # Plot figure
    plt.rcParams.update({'font.size': 18})
    fig, ax = plt.subplots(figsize=(12,10))
    mat = ax.matshow(coefs)

    # Set ticks
    ax.set_xticks(np.arange(190)) 
    ax.set_yticks(np.arange(190))
    ax.set_xticklabels(norm_labels)
    ax.set_yticklabels(norm_labels)
    
    # Show every Nth label
    for i in range(len(norm_labels)):
        if i % 5 != 0:
            norm_labels[i] = ''
        ax.set_xticklabels(norm_labels)
        ax.set_yticklabels(norm_labels)
    fig.colorbar(mat)
    plt.title('Correlation Matrix for {}'.format(component))
    plt.show()
    '''

def linreg_r2scores():
    plt.rcParams.update({'font.size': 18})
    comps = ['co','no','no2','o3','so2','pm2_5','pm10','nh3']
    vals = [0.65, 0.36, 0.56, 0.92, 0.48, 0.49, 0.59, 0.63]
    
    ax = sns.barplot(comps, vals, ci=None, palette='mako')
    sns.despine()
    plt.bar_label(ax.containers[0])
    plt.title('Variance Scores of Unencoded Data')
   
    plt.show()

### Function Calls ###

COMPONENT_NAMES = ['co', 'no', 'no2', 'o3', 'so2', 'pm2_5', 'pm10', 'nh3']
#COMPONENT_NAMES = ['co']
COLORS_LIST = ['tab:blue', 'tab:green', 'tab:orange', 'tab:red', 'tab:purple', 'tab:cyan', 'tab:olive', 'tab:pink']
# Starting dimensions; Change this to edit
DIMS = 190

linegraph('pca', DIMS, COMPONENT_NAMES, COLORS_LIST)
#metrics_comparison('pca', DIMS, COMPONENT_NAMES, COLORS_LIST)
### Correlation Matrix ###
X_DATA_FILE = '/home/nick/github_repos/Pollution-Autoencoders/data/data_norm/co_data_norm.csv'
Y_DATA_FILE = '/home/nick/github_repos/Pollution-Autoencoders/data/vec/vec.csv'
#correlation(X_DATA_FILE, Y_DATA_FILE, 'co')
#linreg_r2scores()
