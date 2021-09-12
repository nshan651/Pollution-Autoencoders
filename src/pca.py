import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv

from sklearn.decomposition import PCA
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import Normalizer
from sklearn.cluster import KMeans


def pca(i, X_train, X_test, Y_train, Y_test, folds, component, dev=False):
    '''
    Perform Linear Regression on either train/validation set or on test set for PCA

    @params:
        i: number of components
        X_train, X_test, Y_train, Y_test: testing/training lists
        dev: set to true if doing validation testing; false if using test set data      
    '''

    # Define the models
    pca = PCA(n_components=i)
    regr = LinearRegression()

    # Encode data
    encoded_X_train = pca.fit_transform(X_train)
    encoded_X_test = pca.transform(X_test)
    
    # Validate the data using an embedding
    print(f'\nLinear regression -- component {component}')
    regr.fit(encoded_X_train, Y_train)
    #res_sum_square = np.mean((regr.predict(encoded_X_test) - Y_test ** 2))
    Y_test_predict = regr.predict(encoded_X_test)

    variance = regr.score(encoded_X_test, Y_test)
    r2 = r2_score(Y_test, Y_test_predict)

    return (variance, r2, encoded_X_test)
    
    # TODO: Remove code below (maybe?)
    '''
    # Linear regression on validation/train set
    if dev == True:
        if i%10 == 0:
            print('PCA validation -- fold {} -- component {}'.format(folds,i))
        train_pca, dev_pca, train_labels_pca, dev_labels_pca = train_test_split(X_train, Y_train, test_size=0.33, random_state=42)
        model.fit(train_pca, train_labels_pca)
        res_sum_square= np.mean((model.predict(dev_pca) - dev_labels_pca ** 2))
        var_val=model.score(dev_pca, dev_labels_pca)
        Y_test_predict = model.predict(dev_pca)
        r2_val=r2_score(dev_labels_pca, Y_test_predict)
        return (model.intercept_, model.coef_, res_sum_square, var_val, r2_val)

    # Linear regression on test set
    else:
        if i%10 == 0:
            print('PCA Test -- component {}'.format(i))
        model.fit(X_train, Y_train)
        res_sum_square= np.mean((model.predict(X_test) - Y_test ** 2))
        var_val=model.score(X_test, Y_test)
        Y_test_predict = model.predict(X_test)
        r2_val=r2_score(Y_test, Y_test_predict)
        return (var_val, r2_val, X_train)
    '''


def pca_train(x, y, folds, dims, component):
    ''' 
    Run PCA using k-fold cross validation strategy

    @params:
        x: Normalized data to encode
        y: Dependent variable
        folds: Number of folds to iterate through
        dims: Number of starting dimensions
        component: The name of the gas/particulate
    '''
    
    # File name
    file_name = f'/home/nick/github_repos/Pollution-Autoencoders/data/test_metrics/pca/{component}_test_metrics'
    # Headers for grid search
    test_metrics_list = ['fold', 'dim', 'variance', 'r2']
    # Write header
    with open(file_name,'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(test_metrics_list)
        f.close() 

    # k-fold cross validation; any cross-validation technique can be substituted here
    kfold = KFold(n_splits=folds, shuffle=True)
    # Number of features to compare
    num_of_comp=list(range(2,dims+1))
    fold_count=0
    # Train/test and metrics for current component's set of data
    train_test_dict = {}
    metrics={}
    # Contains all train_test splits/metrics in PCA for n components and 
    # m sets of splits
    train_test_comp = {}
    metrics_comp = {}
    metrics_comp_set = {}

    print(f'---------- Beginning PCA for gas {component} ----------')
    # Loop through train/test data and save the best data with highest R2 scores
    for training_index, test_index in kfold.split(x):
        # Split X and Y train and test data
        X_train, X_test = x[training_index, :], x[test_index, :]
        Y_train, Y_test = y[training_index], y[test_index]
        fold_count+=1
        # Update dict with train/test values
        train_test_dict['X_train'] = X_train
        train_test_dict['X_test'] = X_test
        train_test_dict['Y_train'] = Y_train
        train_test_dict['Y_test'] = Y_test
        # Save best sets of train/test data that have high R2 scores
        train_test_comp[fold_count] = train_test_dict.copy()
        
        # Train PCA and save a list of metrics 
        for i in num_of_comp:
            variance, r2, _ = pca(i, X_train, X_test, Y_train, Y_test, folds, component, dev=True)
            print(f'fold {fold_count} || dim {i} || variance {variance} || r2 {r2} \n')
            test_metrics_list = [fold_count, i, variance, r2]
            
            # Update test metrics file
            with open(file_name,'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(test_metrics_list)
                f.close() 

            # Create metrics list for current comparison
            #metrics['variance_score']=variance_score
            #metrics['Rsquare']=Rsquare
            #metrics_comp[i] = metrics.copy()
        # Save each metrics comparison for later
        #metrics_comp_set[folds] = metrics_comp.copy()
'''
    # Calculate the R2 score for each component
    best_r2 = -100
    sets = 0
    for i in metrics_comp_set:
        for j in metrics_comp_set[i]:
            R2 = metrics_comp_set[i][j]['Rsquare']
            #print('set : ' + str(i) + ' ' + 'comp' + ' ' + 
            #    str(j) + ' ' + 'Rsquare' + ' ' + str(R2))
            if R2 >= best_r2:
                best_r2 = R2
                sets=i
        


    # Obtain optimal PCA variance and R2 scores and add to list
    for i in train_test_comp:
        if i == sets:
            for j in train_test_comp[sets]:
                if 'X_test'==j:
                    X_ttbest=train_test_comp[sets][j]
                elif 'X_train' ==j:
                    X_tnbest=train_test_comp[sets][j]
                elif 'Y_test' ==j:
                    Y_ttbest=train_test_comp[sets][j]
                elif 'Y_train'==j:
                    Y_tnbest=train_test_comp[sets][j]
    
    # Graph PCA to test set with best components
    # using the best set to call PCA function to use the test set with all components
    pca_r2=[]
    pca_variance=[]
    
    # Test set for pca
    for i in num_of_comp:   
        variance, R2, X_train = pca(i, X_tnbest, X_ttbest, Y_tnbest ,Y_ttbest, folds, dev=False)
        pca_variance.append(variance)
        pca_r2.append(R2)
        #pca_X_train.append(X_train)
            
    # Normalize var and r2 so they are same length as ax1 and ax2
    append_size = len(X_train[:,0]) - dims + 1
    norm = np.empty(append_size)
    norm[:] = np.NaN
    var_norm = [*pca_variance, *norm]
    r2_norm = [*pca_r2, *norm]

    # Output dict to write 
    output_dict = {
        '{}_X_train_ax1'.format(component): X_train[:,0],
        '{}_X_train_ax2'.format(component) : X_train[:,1],
        '{}_var'.format(component) : var_norm,
        '{}_r2'.format(component) : r2_norm
    }
    
    # Write entry to file
    file_name = '/home/nicks/github_repos/Pollution-Autoencoders/data/model_results/pca/{}_pca_results.csv'.format(component)
    write_data = pd.DataFrame(data=output_dict)
    write_data.to_csv(path_or_buf=file_name, index=False)
'''


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


def pca_run(x, y, splits, dims, component):
    ''' 
    Run PCA using k-fold cross validation strategy

    @params:
        x: Normalized data to encode
        y: Dependent variable
        dims: Number of starting dimensions
        component: The names of the gases/particulate
    '''

    # Retrieve list of city names to append
    df = pd.read_csv(f'home/nick/github_repos/Pollution-Autoencoders/data/data_clean/{component}_data_clean.csv')
    cities = df['city'].values
    
    # Write headers for model metrics and vecs; erase previous model
    metrics_file = f'/home/nick/github_repos/Pollution-Autoencoders/data/model_metrics/pca/{component}_metrics.csv'
    vec_file = f'/home/nick/github_repos/Pollution-Autoencoders/data/vec/pca/{component}_vec.csv'
    
    # Number of features to compare
    num_of_dims=list(range(1,dims+1))

    vector = []
    var_list = []
    r2_list = []
    print(f'---------- Beginning PCA for gas {component} ----------')
    for i in num_of_dims:
        print(f'---------- Component {i} ----------')
        # Define the models
        pca = PCA(n_components=i)
        regr = LinearRegression()
        
        # Encode data
        #X_train = pca.fit_transform(X_train)
        encoded_data = pca.transform(x)

        # Validate the embedding by performing a linear regression
        #res_sum_square = np.mean((regr.predict(x) - y ** 2))
        regr.fit(encoded_data, y)
        
        Y_test_predict = regr.predict(encoded_data)

        # Variance and r2 scores
        variance = regr.score(encoded_data, y)
        r2 = r2_score(y, Y_test_predict)
        # Save values for each comp dim
        var_list.append(variance)
        r2_list.append(r2)
        vector = list(encoded_data)
        print (f'Variance score: {variance} for component {i}')
        print (f'R Square {r2} for component {i}')

    # Write all vector data 
    vec_labels = [f'dim_{i}' for i in range(1, dims+1)]
    vector_data = pd.DataFrame(data=vector, columns=vec_labels)
    # Add city labels
    vector_data.insert(0, 'city', cities)
    vector_data.to_csv(path_or_buf=vec_file, index=None)
    
    # Write variance and r2 scores of each gas for every dimension 
    output_dict = {'dim': num_of_dims, 'variance' : var_list, 'r2' : r2_list}
    metrics_data = pd.DataFrame(data=output_dict)
    metrics_data.to_csv(path_or_buf=metrics_file, index=False)
        
    
def pca_kmeans(component, colors_list):
    ''' kmeans clustering on a single gas '''

    plt.figure(figsize=(10,10))
    plt.rcParams.update({'font.size': 18})
    # Run PCA
    *_, X_train = pca_run('co')
    # Perform KMeans
    kmeans = KMeans(n_clusters=3)
    # Compute cluster centers and predict cluster indicies
    X_cluster = kmeans.fit_predict(X_train)

    label_color = [colors_list[i] for i in X_cluster]
    plt.scatter(X_train[:,0], X_train[:,2], c=label_color, alpha=0.5)
    plt.title('K-means Clustering of CO')
    plt.show()


def main():
    ''' Set up sources, call functions as needed '''

    # Constants
    #component_names = ['co', 'no', 'no2', 'o3', 'so2', 'pm2_5', 'pm10', 'nh3']
    test_component = 'co'
    colors_list = ['tab:blue', 'tab:green', 'tab:orange', 'tab:red', 'tab:purple', 'tab:cyan', 'tab:olive', 'tab:pink']
    dims = 190
    folds = 5
    # Open normalized data; y value list using last day of 7-month data
    dfx = pd.read_csv(f'/home/nick/github_repos/Pollution-Autoencoders/data/data_norm/{test_component}_data_norm.csv')
    dfy = pd.read_csv(f'/home/nick/github_repos/Pollution-Autoencoders/data/data_clean/{test_component}_data_clean.csv')
    # Set x as the normalized values, y as the daily average of final day
    x = dfx.values
    y = dfy.loc[:, ['{}_2021_06_06'.format(test_component)]].values

    ### Function calls ###
    interpolate('co')
    #pca_train(x, y, folds, dims, test_component)
    #pca_run(dims, component)
    

if __name__ == '__main__':
    main()

