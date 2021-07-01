import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import Normalizer
from sklearn.cluster import KMeans


def pca(i, X_train, X_test, Y_train, Y_test, folds=0, dev=False):
    '''
    Perform Linear Regression on either train/validation set or on test set for PCA

    @params:
        i: number of components
        X_train, X_test, Y_train, Y_test: testing/training lists
        dev: set to true if doing validation testing; false if using test set data      
    '''

    # Perform PCA on data set
    pca = PCA(n_components=i)
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)
    model = LinearRegression()

    # Linear regression on validation/train set
    if dev == True:
        if i%10 == 0:
            print('PCA validation -- fold {} -- component {}'.format(folds,i))
        train_pca, dev_pca, train_labels_pca, dev_labels_pca = train_test_split(X_train, Y_train, test_size=0.33, random_state=42)
        model.fit(train_pca, train_labels_pca)
        res_sum_square= np.mean((model.predict(dev_pca) - dev_labels_pca ** 2))
        var_val=model.score(dev_pca, dev_labels_pca)
        Y_test_predict = model.predict(dev_pca)
        r2_val=r2_score(dev_labels_pca,Y_test_predict)
        return (model.intercept_, model.coef_, res_sum_square, var_val, r2_val)

    # Linear regression on test set
    else:
        if i%10 == 0:
            print('PCA Test -- component {}'.format(i))
        model.fit(X_train, Y_train)
        res_sum_square= np.mean((model.predict(X_test) - Y_test ** 2))
        var_val=model.score(X_test, Y_test)
        Y_test_predict = model.predict(X_test)
        r2_val=r2_score(Y_test,Y_test_predict)
        return (var_val, r2_val, X_train)


def pca_run(dims, component_names):
    ''' 
    Run PCA using k-fold cross validation strategy

    @params:
        dims: Number of starting dimensions
        component_names: The names of the gases/particulates
    '''

    for component in component_names:
        print('---------- Beginning PCA for {} ----------'.format(component))
        ### Preprocessing ###
        df = pd.read_csv('/home/nicks/github_repos/Pollution-Autoencoders/data/data_clean/{}_data_clean.csv'.format(component))
            
        # Features list and removal of city, lat, lon
        features = list(df.columns.values)
        del features[:1]    
        del features[-1]

        # y value list using last day of 7-month data
        y = df.loc[:, ['{}_2021_06_06'.format(component)]].values

        # Normalize x values
        x = df.loc[:, features].values
        x = Normalizer().fit_transform(x)

        # Number of features to compare
        num_of_comp=list(range(2,dims+1))

        ### Cross Validation ### 

        # k-fold cross validation; any cross-validation technique can be substituted here
        kfold = KFold(n_splits=2, shuffle=True)
        folds=0
        # Train/test and metrics for current component's set of data
        train_test_dict = {}
        metrics={}
        # Contains all train_test splits/metrics in PCA for n components and 
        # m sets of splits
        train_test_comp = {}
        metrics_comp = {}
        metrics_comp_set = {}

        # Loop through train/test data and save the best data with highest R2 scores
        for training_index, test_index in kfold.split(x):
            # Split X and Y train and test data
            X_train, X_test = x[training_index, :], x[test_index, :]
            Y_train, Y_test = y[training_index], y[test_index]
            folds+=1
            # Update dict with train/test values
            train_test_dict['X_train'] = X_train
            train_test_dict['X_test'] = X_test
            train_test_dict['Y_train'] = Y_train
            train_test_dict['Y_test'] = Y_test
            # Save best sets of train/test data that have high R2 scores
            train_test_comp[folds] = train_test_dict.copy()
            
            # Train PCA and save a list of metrics 
            for i in num_of_comp:
                model_intercept, model_coef, res_sum_square, variance_score, Rsquare = pca(i, X_train, X_test, Y_train, Y_test, folds, dev=True)
                # Create metrics list for current comparison
                metrics['model_intercept']=model_intercept
                metrics['model_coef']=model_coef
                metrics['res_sum_square']=res_sum_square
                metrics['variance_score']=variance_score
                metrics['Rsquare']=Rsquare
                metrics_comp[i] = metrics.copy()
            # Save each metrics comparison for later
            metrics_comp_set[folds] = metrics_comp.copy()

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
            
        # Graph PCA to test set with best components
        # using the best set to call PCA function to use the test set with all components
        pca_r2=[]
        pca_variance=[]

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
        
    
def pca_kmeans(component, colors_list):
    '''
    kmeans clustering on a single gas
    '''

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


### RUN ###

COMPONENT_NAMES = ['co', 'no', 'no2', 'o3', 'so2', 'pm2_5', 'pm10', 'nh3']
#COMPONENT_NAMES = ['co']
COLORS_LIST = ['tab:blue', 'tab:green', 'tab:orange', 'tab:red', 'tab:purple', 'tab:cyan', 'tab:olive', 'tab:pink']
DIMS = 190

pca_run(DIMS, COMPONENT_NAMES)

