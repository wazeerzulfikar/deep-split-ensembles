import pandas as pd
from sklearn.datasets import load_boston
import glob, os, math, time, re, csv
from decimal import *
import numpy as np
import pandas as pd
from sklearn.datasets import load_boston
np.random.seed(0)

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import scale

from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import pdist


def load_dataset(config):

    if config.dataset=='boston':
        data = _boston(config)        

    elif config.dataset=='cement':
        data = _cement(config)

    # elif config.dataset=='energy':
    #     data = _cement(config)

    # elif config.dataset=='cement':
    #     data = _cement(config)

    return data

def random_split(features):
    data = np.transpose(features)
    n_splits = feature_split(features, return_split_sizes=True)
    n_features = len(data)
    
    per_split = n_features//n_splits

    # this just makes sure that number of splits is equal to the splits in hierarchial
    # and also that the "extra" features are equally distributed amongst the clusters  
    extra_cols = n_features%n_splits  
    
    rand_range = [x for x in range(n_features)]
    np.random.shuffle(rand_range)
    X = []
    ind=0
    while ind<n_features:
        if ind+per_split>=n_features:
            f = rand_range[ind:]    
        else:
            f = rand_range[ind:ind+per_split+int(extra_cols>0)]
            if(extra_cols>0):
                ind+=1
                extra_cols-=1
        X.append(np.transpose(data[f]))
        ind+=per_split
    return X

def feature_split(features, return_split_sizes=False):
    from scipy.cluster.hierarchy import linkage
    data = np.transpose(features)

    ######### Hierarchical Clustering based on correlation
    Y = pdist(data, 'correlation')
    linkage = linkage(Y, 'complete')
    clusters = fcluster(linkage, 0.5 * Y.max(), 'distance')
    if(return_split_sizes):
        return len(set(clusters))

    X = []
    for cluster in set(clusters):
        indices = [j for j in range(len(clusters)) if clusters[j]==cluster]
        X.append(np.transpose(data[indices]))
    return X

def _boston(config):
    data_df = load_boston()
    df = pd.DataFrame(data=data_df['data'], columns=data_df['feature_names'])
    y = data_df['target']

    if config.mod_split=='none':
        X = df.values
        data = {'X':X, 'y':y}

    elif config.mod_split=='human':
        features1 = ['ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX']
        features2 = ['CRIM', 'PTRATIO', 'B', 'LSTAT']
        X1 = df[features1].values
        X2 = df[features2].values
        data = {'X1':X1, 'X2':X2, 'y':y}

    elif config.mod_split=='random':
        X = random_split(df.values)
        data = {'y':y}
        for i, x in enumerate(X):
            data[str(i)] = x

    elif config.mod_split=='computation_split':
        X = feature_split(df.values)
        data = {'y':y}
        for i, x in enumerate(X):
            data[str(i)] = x

    return data

def _cement(config):
    data_df = pd.read_csv(os.path.join(config.regression_datasets_dir, 'cement.csv'))
    
    target_col = 'Concrete compressive strength(MPa, megapascals) '
    y = data_df[target_col]

    df = data_df.drop(columns=[target_col])
    
    if config.mod_split=='none' or config.mod_split=='human': # since only 1 split
        X = df.values
        data = {'X':X, 'y':y}

    elif config.mod_split=='random':
        X = random_split(df.values)
        data = {'y':y}
        for i, x in enumerate(X):
            data[str(i)] = x

    elif config.mod_split=='computation_split':
        X = feature_split(df.values)
        data = {'y':y}
        for i, x in enumerate(X):
            data[str(i)] = x

    return data
