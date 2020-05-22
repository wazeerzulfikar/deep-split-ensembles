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

    elif config.dataset=='energy_efficiency':
        data = _energy_efficiency(config)

    elif config.dataset=='kin8nm':
        data = _kin8nm(config)

    elif config.dataset=='power_plant':
        data = _power_plant(config)

    elif config.dataset=='protein':
        data = _protein(config)

    elif config.dataset=='wine':
        data = _wine(config)

    return data

def random_split(features):
    data = np.transpose(features)
    clusters = feature_split(features, return_split_sizes=True)
    n_features = len(data)
    
    rand_range = [x for x in range(n_features)]
    np.random.shuffle(rand_range)
    
    X = []
    ind=0
    
    for c in set(clusters):
        cluster_size = list(clusters).count(c)
        indices = rand_range[ind:ind+cluster_size]
        X.append(np.transpose(data[indices]))
        ind+=cluster_size
    return X

def feature_split(features, return_split_sizes=False):
    from scipy.cluster.hierarchy import linkage
    data = np.transpose(features)

    ######### Hierarchical Clustering based on correlation
    Y = pdist(data, 'correlation')
    linkage = linkage(Y, 'complete')
    # dendrogram(linkage, color_threshold=0)
    # plt.show()
    clusters = fcluster(linkage, 0.5 * Y.max(), 'distance')
    if(return_split_sizes):
        return clusters

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
        data = {'0':X, 'y':y}

    elif config.mod_split=='human':
        features1 = ['ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX']
        features2 = ['CRIM', 'PTRATIO', 'B', 'LSTAT']
        X1 = df[features1].values
        X2 = df[features2].values
        data = {'0':X1, '1':X2, 'y':y}

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
        data = {'0':X, 'y':y}

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

def _energy_efficiency(config):
    return

def _kin8nm(config):
    data_df = pd.read_csv(os.path.join(config.regression_datasets_dir, 'kin8nm.csv'))
    
    y = data_df['y']

    df = data_df.drop(columns=['y'])
    if config.mod_split=='none' or config.mod_split=='human': # since only 1 split
        X = df.values
        data = {'0':X, 'y':y}

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

def _power_plant(config):
    data_df = pd.read_csv(os.path.join(config.regression_datasets_dir, 'power_plant.csv'))
    
    y = data_df['PE']

    df = data_df.drop(columns=['PE'])
    if config.mod_split=='none':
        X = df.values
        data = {'0':X, 'y':y}

    elif config.mod_split=='human':
        features1 = ['AT', 'RH']
        features2 = ['V', 'AP']
        X1 = df[features1].values
        X2 = df[features2].values
        data = {'0':X1, '1':X2, 'y':y}

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

def _protein(config):
    data_df = pd.read_csv(os.path.join(config.regression_datasets_dir, 'protein.csv'))
    
    y = data_df['RMSD']

    df = data_df.drop(columns=['RMSD'])

    if config.mod_split=='none' or config.mod_split=='human':
        X = df.values
        data = {'0':X, 'y':y}

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

def _wine(config):
    data_df = pd.read_csv(os.path.join(config.regression_datasets_dir, 'wine.csv'))
    
    y = data_df['quality']

    df = data_df.drop(columns=['quality'])

    if config.mod_split=='none' or config.mod_split=='human':
        X = df.values
        data = {'0':X, 'y':y}

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
