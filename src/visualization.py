import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import pyparsing
import mplcursors
from mpl_toolkits import mplot3d 
import plotly.express as px


def linegraph(f_name, dims, component_names, colors_list):
    ''' 
    Plot the explained variance across dimensions using the model results

    @params:
        f_name: Name of the file
        type: Type of reduction method, pca or autoencoders
        dims: Number of dimensions
        component_names: Gas/particulate list
        colors_list: List of colors to be used
    '''
    
    num_of_dims = list(range(2, dims+1))
    # Plot results
    plt.figure(figsize=(12,10))
    for i, component in enumerate(component_names):
        file_name = f'{f_name}{component}_metrics.csv'
        plt_title = 'Autoencoder Reduced Representation of Air Pollutants'

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
            high_file = f'/home/nick/git/Pollution-Autoencoders/data/modle_metrics/ae/best_worst/{component}_best_metrics.csv'
            low_file = f'/home/nick/git/Pollution-Autoencoders/data/model_metrics/ae/best_worst/{component}_worst_metrics.csv'
            plt_title = 'Autoencoder Reduced Representation of Air Pollutants'
        elif type == 'pca':
            high_file = f'/home/nick/git/Pollution-Autoencoders/data/model_metrics/pca/best_worst/{component}_best_metrics.csv'
            low_file = f'/home/nick/git/Pollution-Autoencoders/data/model_metrics/pca/best_worst/{component}_worst_metrics.csv'
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


def heatmap(component):
    # Read in component data and create an annotation list of cities
    data = pd.read_csv('/home/nicks/git/Pollution-Autoencoders/data/data_clean/{}_data_clean.csv'.format(component))
    df = pd.DataFrame(data=data)
    annotations = df['city']

    # Read in whitelist of cities to graph
    wlist_data = pd.read_csv(filepath_or_buffer='/home/nick/git/Pollution-Autoencoders/data/other/outliers_whitelist.csv')
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
   
    X_matrix = []
    norm_labels = ['dim_{}'.format(i) for i in range(1, SIZE+1)]
    
    
    for i in range(SIZE):
        X_matrix.append(norm_data[norm_labels[i]][:])
    print('X_matrix', X_matrix) 
    coefs = np.corrcoef(X_matrix)

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
        square=True,
        cbar_kws={'label': 'Correlation strength'}
    )
    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=45,
        horizontalalignment='right'
    )
    #ax.set_xlabel('dim_1')
    #ax.set_ylabel('dim_2')
    plt.show()
    

def scatter(component, title, mask=False, hover=False):
    '''
    Autoencoder scatter plot of multiple components for the first two dimensions

    @params:
        component: Name of gas or particulate
        title: Part of the graph title 
        mask: Flag to apply a custom color mask (defaults to pre-made color gradient)
        hover: Flag to apply a hover effect to the graph (defaults to false)
    '''

    plt.rcParams.update({'font.size': 16})
    df = pd.read_csv(f'/home/nick/git/Pollution-Autoencoders/data/vec/{component}_vec_4.csv')
    top200_data = pd.read_csv(f'/home/nick/git/Pollution-Autoencoders/data/other/top200.csv')
    top200 = [i for i in top200_data['city']]
    # whitelist labels
    wlist = ['RedWing', 'NewYork', 'LosAngeles', 'Chicago']
    # Region lists
    west = ['CA', 'OR', 'WA', 'MT', 'OH', 'NV', 'AZ', 'UT', 'NM', 'WY', 'CO', 'AK', 'HI', 'ID']
    midwest = ['IL', 'IN', 'MI', 'OH', 'WI', 'IA', 'KS', 'MN', 'MO', 'NE', 'ND', 'SD']
    south = ['DE', 'FL', 'GA', 'MD', 'NC', 'SC', 'VA', 'WV', 'AL', 'KY', 'MS','TN','AR','LA', 'OK', 'TX', 'DC']
    north_east = ['PA', 'NY', 'RI', 'MD', 'CT', 'NJ', 'MA', 'VT', 'NH', 'ME']
    # Encoded dims
    x = df['dim_3']
    y = df['dim_4']

    fig, ax = plt.subplots(figsize = (10,7))

    if mask:
        # Initialize color masking
        cmask = {'State cities': {'x' : [], 'y' : []},
            'Other cities' : {'x' : [], 'y' : []}
        }
        '''
        cmask = {'West': {'x' : [], 'y' : []},
            'Northeast': {'x' : [], 'y' : []},
            'South' : {'x' : [], 'y' : []}, 
            'Midwest' : {'x' : [], 'y' : []}, 
            'Outside US' : {'x' : [], 'y' : []}
        }
        '''
        temp = 0 
        # Create color mask based on conditions
        for i in range(len(x)):
            #if df['state'][i] == 'AZ':
            if df['state'][i] not in (west+midwest+south+north_east):
                #if df['country'][i] != 'PR':
                    #print(df['country'][i])
                temp+=1 
            if df['city'][i] in top200:
                cmask['State cities']['x'].append(x[i])
                cmask['State cities']['y'].append(y[i])
            else:
                cmask['Other cities']['x'].append(x[i])
                cmask['Other cities']['y'].append(y[i])

            '''
            if df['state'][i] in west:
                #cmask['Cities in US']['x'].append(x[i])
                #cmask['Cities in US']['y'].append(y[i])
                cmask['West']['x'].append(x[i])
                cmask['West']['y'].append(y[i])
            elif df['state'][i] in north_east:
                #cmask['Cities in US']['x'].append(x[i])
                #cmask['Cities in US']['y'].append(y[i])
                cmask['Northeast']['x'].append(x[i])
                cmask['Northeast']['y'].append(y[i])
            elif df['state'][i] in south: 
                #cmask['Cities in US']['x'].append(x[i])
                #cmask['Cities in US']['y'].append(y[i])
                cmask['South']['x'].append(x[i])
                cmask['South']['y'].append(y[i])
            elif df['state'][i] in midwest:
                #cmask['Cities in US']['x'].append(x[i])
                #cmask['Cities in US']['y'].append(y[i])
                cmask['Midwest']['x'].append(x[i])
                cmask['Midwest']['y'].append(y[i])
            else:
                cmask['Outside US']['x'].append(x[i])
                cmask['Outside US']['y'].append(y[i])
            '''
            ''' 
               #print(f'{df["city"][i]},{df["state"][i]}')
                #if i%20==0:
                #    ann = ax.annotate(text=f"{df['city'][i]},{df['state'][i]}", xy=(x[i],y[i]))
            ''' 
        '''
        # Create scatter based on custom color mask
        ax.scatter(cmask['West']['x'], cmask['West']['y'], color='blue', alpha=0.3)
        ax.scatter(cmask['Northeast']['x'], cmask['Northeast']['y'], color='yellow', alpha=0.3)
        ax.scatter(cmask['South']['x'], cmask['South']['y'], color='red', alpha=0.3)
        ax.scatter(cmask['Midwest']['x'], cmask['Midwest']['y'], color='green', alpha=0.3)
        ax.scatter(cmask['Outside US']['x'], cmask['Outside US']['y'], color='gray', alpha=0.3)
        '''
        ax.scatter(cmask['Other cities']['x'], cmask['Other cities']['y'], color='gray', alpha=0.3)
        ax.scatter(cmask['State cities']['x'], cmask['State cities']['y'], color='orange', alpha=0.5)

    else:
        # Else, use a pre-made color map
        cmap = plt.get_cmap('hot')
        sct = ax.scatter(x, y, cmap=cmap, c=(x+y), alpha=0.5)
        fig.colorbar(sct, ax = ax, shrink = 0.5, aspect = 5)
    # Mouse-over annotations
    if hover:
        # Annotate points
        dlabel = []
        for city, state in zip(df['city'], df['state']):
            dlabel.append(f'{city},{state}')
        # Cursor Object
        cursor = mplcursors.cursor(hover=True)
        cursor.connect('add', lambda sel: sel.annotation.set_text(dlabel[sel.target.index]))

    '''
    # Full annotations
    elif mode == 'full':
        for i, (city, state) in enumerate(zip(df['city'], df['state'])):
            if i%200==0:
                ann = ax.annotate(text=f'{city},{state}', xy=(x[i],y[i]))
        ann.set_visible(False)
    
    else:
        raise Exception('Mode is not valid')
    '''
    ax.set_xlabel('dim_3')
    ax.set_ylabel('dim_4')
    sns.despine(ax=ax, offset=0)
    plt.title(f'Dimensions 3 and 4 of {component} {title}')
    #ax.legend(['West', 'Northeast', 'South', 'Midwest', 'Outside US'])
    ax.legend(['Other cities','Top 200 most populous cities'])
    #plt.show()
    print(f'number of cities in PR {temp}')


def scatter3D(component):
    plt.rcParams.update({'font.size': 14})
    df = pd.read_csv(f'./data/vec/{component}_vec_3.csv')
    
    # whitelist labels
    wlist = ['RedWing', 'NewYork', 'LosAngeles', 'Chicago']
    #wlist = ['RedWing']
    # 3 encoded dims
    x = df['dim_1']
    y = df['dim_2']
    z = df['dim_3']

    fig = plt.figure(figsize = (10,7))
    ax = plt.axes(projection = '3d')

    # Creating color map
    cmap = plt.get_cmap('hot')
    #ax.scatter3D(df['dim_1'], df['dim_2'], df['dim_3'], color = 'g')
    sct = ax.scatter3D(x, y, z, cmap=cmap, c=(x+y+z), alpha=0.4)
    ax.set_xlabel('dim_1')
    ax.set_ylabel('dim_2')
    ax.set_zlabel('dim_3')
    fig.colorbar(sct, ax = ax, shrink = 0.5, aspect = 5)
    plt.title(f'3D Scatter of latent dims for {component}')

    # Ad hoc annotations
    for i, city in enumerate(df['city']):
        for target in wlist:
            if city == target:
                print(city)
                plt.annotate(text=city, xy=(x[i],y[i]))
    plt.show()


def linreg_r2scores():
    plt.rcParams.update({'font.size': 18})
    comps = ['co','no','no2','o3','so2','pm2_5','pm10','nh3']
    vals = [0.65, 0.36, 0.56, 0.92, 0.48, 0.49, 0.59, 0.63]
    
    ax = sns.barplot(comps, vals, ci=None, palette='mako')
    sns.despine()
    plt.bar_label(ax.containers[0])
    plt.title('Variance Scores of Unencoded Data')
   
    plt.show()


def scattergeo(component):
    df = pd.read_csv(f'/home/nick/git/Pollution-Autoencoders/data/data_clean/{component}_data_clean.csv')
    df2 = pd.read_csv(f'/home/nick/git/Pollution-Autoencoders/data/vec/{component}_vec_2.csv')
    mcolor = df2['dim_2']
    #mcolor = df['o3_2021_06_06']
    #mcolor = df['o3_2020_11_27']
    #bar_title = 'Embedding Value'
    #bar_title = 'μg/m3'
    #title = 'O3 First Encoded Dimension'
    #title = 'O3 Pollution Values 2021.06.06'
    fig = px.scatter_geo(
        data_frame=df,
        lat = df['lat'],
        lon = df['lon'],
        color=mcolor,
        color_continuous_scale='portland'
    )

    fig.layout.coloraxis.colorbar.title = ''
    fig.layout.coloraxis.colorbar.tickfont.size = 18
    #fig.update_coloraxes(showscale=False)

    fig.update_layout(
        #title = title,
        geo_scope='usa'
    )
    
    fig.show()

### Function Calls ###

#COMPONENT_NAMES = ['co', 'no', 'no2', 'o3', 'so2', 'pm2_5', 'pm10', 'nh3']
COMPONENT_NAMES = ['co', 'no', 'no2']
#COMPONENT_NAMES = ['co']
COLORS_LIST = ['tab:blue', 'tab:green', 'tab:orange', 'tab:red', 'tab:purple', 'tab:cyan', 'tab:olive', 'tab:pink']
# Starting dimensions; Change this to edit
DIMS = 190

#linegraph('/home/nick/git/Pollution-Autoencoders/data/model_metrics/ae/', DIMS, COMPONENT_NAMES, COLORS_LIST)
#metrics_comparison('pca', DIMS, COMPONENT_NAMES, COLORS_LIST)
### Correlation Matrix ###
X_DATA_FILE = '/home/nick/git/Pollution-Autoencoders/data/data_norm/co_data_norm.csv'
Y_DATA_FILE = '/home/nick/git/Pollution-Autoencoders/data/vec/vec.csv'
correlation(X_DATA_FILE, Y_DATA_FILE, 'co')
#linreg_r2scores()
#scatter3D('o3')
#scatter('o3', 'for top 200 most populous cities in US', mask=True, hover=False)
#scattergeo('o3')
