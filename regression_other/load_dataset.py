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

    elif config.dataset=='energy':
        data = _cement(config)

    elif config.dataset=='cement':
        data = _cement(config)

    return data

def feature_split(features):
    from scipy.cluster.hierarchy import linkage
    data = np.transpose(features)

    ######### Hierarchical Clustering based on correlation
    Y = pdist(data, 'correlation')
    linkage = linkage(Y, 'complete')
    clusters = fcluster(linkage, 0.5 * Y.max(), 'distance')

    X = []
    for cluster in set(clusters):
        indices = [j for j in range(len(clusters)) if clusters[j]==cluster]
        X.append(np.transpose(data[indices]))
    return X

def _boston(config):
    print("Loading boston")
    data_df = load_boston()
    df = pd.DataFrame(data=data_df['data'], columns=data_df['feature_names'])
    df['TARGET'] = data_df['target']

    if config.mod_split=='none':
        X = df.values
        y = df['target']
        data = {'X':X, 'y':y}

    elif config.mod_split=='human':
        features1 = ['ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX']
        features2 = ['CRIM', 'PTRATIO', 'B', 'LSTAT']
        X1 = df[features1].values
        X2 = df[features2].values
        y = df['TARGET']
        data = {'X1':X1, 'X2':X2, 'y':y}

    # elif config.mod_split=='random':


    elif config.mod_split=='computation_split':
        X = feature_split((df.drop(columns=['TARGET'])).values)
        y = df['TARGET']
        data = {'y':y}
        for i, x in enumerate(X):
            data[str(i)] = x

    return data

def _cement(config):
    print("Loading cement")
    data=0
    return data