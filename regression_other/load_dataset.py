import pandas as pd
from sklearn.datasets import load_boston
import glob, os, math, time, re, csv
from decimal import *
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.datasets import load_boston
np.random.seed(0)

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import scale

from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import pdist

def random_split(config, features):
    data = np.transpose(features)
    clusters = feature_split(config, features, return_split_sizes=True)
    n_features = len(data)
    
    rand_range = [x for x in range(n_features)]
    np.random.shuffle(rand_range)
    
    X = []
    ind=0
    
    for c in set(clusters):
        cluster_size = list(clusters).count(c)
        indices = rand_range[ind:ind+cluster_size]
        # print(indices)
        X.append(np.transpose(data[indices]))
        ind+=cluster_size
    return X

def feature_split(config, features, return_split_sizes=False):
    from scipy.cluster.hierarchy import linkage
    data = np.transpose(features)

    ######### Hierarchical Clustering based on correlation
    Y = pdist(data, 'correlation')
    linkage = linkage(Y, 'complete')
    # dendrogram(linkage, color_threshold=0)
    # plt.show()
    if config.dataset=='msd':
        clusters = fcluster(linkage, 0.75 * Y.max(), 'distance')
    else:
        clusters = fcluster(linkage, 0.5 * Y.max(), 'distance')

    if(return_split_sizes):
        return clusters

    X = []
    for cluster in set(clusters):
        indices = [j for j in range(len(clusters)) if clusters[j]==cluster]
        # print(indices)
        X.append(np.transpose(data[indices]))
    return X

def load_dataset(config):
    np.random.seed(0)
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

    elif config.dataset=='yacht':
        data = _yacht(config)
    
    elif config.dataset=='naval':
        data = _naval(config)

    elif config.dataset=='msd':
        data = _msd(config)

    elif config.dataset=='life':
        data = _life(config)
    return data



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
        X = random_split(config, df.values)
        data = {'y':y}
        for i, x in enumerate(X):
            data[str(i)] = x

    elif config.mod_split=='computation_split':
        X = feature_split(config, df.values)
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
        X = random_split(config, df.values)
        data = {'y':y}
        for i, x in enumerate(X):
            data[str(i)] = x

    elif config.mod_split=='computation_split':
        X = feature_split(config, df.values)
        data = {'y':y}
        for i, x in enumerate(X):
            data[str(i)] = x

    return data

def _energy_efficiency(config):

    data_df = pd.read_csv(os.path.join(config.regression_datasets_dir, 'energy_efficiency.csv'))
    
    target_col1 = 'Heating Load'
    target_col2 = 'Cooling Load'
    y = data_df[target_col1]

    df = data_df.drop(columns=[target_col1, target_col2])

    if config.mod_split=='none':
        X = df.values
        data = {'0':X, 'y':y}

    if config.mod_split=='human':
        features1 = ['Wall Area', 'Roof Area', 'Glazing Area', 'Glazing Area Distribution']
        features2 = ['Relative Compactness', 'Surface Area', 'Overall Height', 'Orientation']

        X1 = df[features1].values
        X2 = df[features2].values
        data = {'0':X1, '1':X2, 'y':y}

    elif config.mod_split=='random':
        X = random_split(config, df.values)
        data = {'y':y}
        for i, x in enumerate(X):
            data[str(i)] = x

    elif config.mod_split=='computation_split':
        X = feature_split(config, df.values)
        data = {'y':y}
        for i, x in enumerate(X):
            data[str(i)] = x

    return data

def _kin8nm(config):
    data_df = pd.read_csv(os.path.join(config.regression_datasets_dir, 'kin8nm.csv'))
    
    y = data_df['y']

    df = data_df.drop(columns=['y'])
    if config.mod_split=='none' or config.mod_split=='human': # since only 1 split
        X = df.values
        data = {'0':X, 'y':y}

    elif config.mod_split=='random':
        X = random_split(config, df.values)
        data = {'y':y}
        for i, x in enumerate(X):
            data[str(i)] = x

    elif config.mod_split=='computation_split':
        X = feature_split(config, df.values)
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
        X = random_split(config, df.values)
        data = {'y':y}
        for i, x in enumerate(X):
            data[str(i)] = x

    elif config.mod_split=='computation_split':
        X = feature_split(config, df.values)
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
        X = random_split(config, df.values)
        data = {'y':y}
        for i, x in enumerate(X):
            data[str(i)] = x

    elif config.mod_split=='computation_split':
        X = feature_split(config, df.values)
        data = {'y':y}
        for i, x in enumerate(X):
            data[str(i)] = x

    return data

def _wine(config):
    data_df = pd.read_csv(os.path.join(config.regression_datasets_dir, 'wine.csv'))
    
    y = data_df['quality']

    df = data_df.drop(columns=['quality'])

    if config.mod_split=='none':
        X = df.values
        data = {'0':X, 'y':y}

    if config.mod_split=='human':
        features1 = ['fixed acidity', 'volatile acidity', 'density', 'pH']
        features2 = ['citric acid', 'residual sugar', 'chlorides','free sulfur dioxide',
                    'total sulfur dioxide', 'sulphates', 'alcohol']

        X1 = df[features1].values
        X2 = df[features2].values
        data = {'0':X1, '1':X2, 'y':y}

    elif config.mod_split=='random':
        X = random_split(config, df.values)
        data = {'y':y}
        for i, x in enumerate(X):
            data[str(i)] = x

    elif config.mod_split=='computation_split':
        X = feature_split(config, df.values)
        data = {'y':y}
        for i, x in enumerate(X):
            data[str(i)] = x

    return data

def _yacht(config):
    cols = ['Longitudinal position of the center of buoyancy', 'Prismatic coefficient', 'Length-displacement ratio', 
            'Beam-draught ratio', 'Length-beam ratio', 'Froude number', 'Residuary resistance per unit weight of displacement']

    data_df = pd.read_csv(os.path.join(config.regression_datasets_dir, 'yacht.data'), sep="\\s+", names=cols)
    target_col = 'Residuary resistance per unit weight of displacement'
    y = data_df[target_col]
    df = data_df.drop(columns=[target_col])
    if config.mod_split=='none' or config.mod_split=='human':
        X = df.values
        data = {'0':X, 'y':y}

    elif config.mod_split=='random':
        X = random_split(config, df.values)
        data = {'y':y}
        for i, x in enumerate(X):
            data[str(i)] = x

    elif config.mod_split=='computation_split':
        X = feature_split(config, df.values)
        data = {'y':y}
        for i, x in enumerate(X):
            data[str(i)] = x

    return data

def _naval(config):
    cols = ['lp', 'v', 'gtt', 'gtn', 'ggn', 'ts', 'tp',
            't48', 't1', 't2', 'p48', 'p1', 'p2', 'pexh',
            'tic', 'mf', 'y1', 'y2']

    data_df = pd.read_csv(os.path.join(config.regression_datasets_dir, 'naval.csv'), sep='\\s+', names=cols)
    target_col = ['y1', 'y2']
    
    y = data_df[target_col[0]]

    df = data_df.drop(columns=['y1', 'y2', 't1', 'p1'])

    if config.mod_split=='none':
        X = df.values
        data = {'0':X, 'y':y}

    if config.mod_split=='human':
        features1 = ['t48', 't2']
        features2 = ['p48', 'p2', 'pexh']
        features3 = ['gtt', 'gtn', 'ggn', 'ts', 'tp']
        features4 = ['lp', 'v', 'tic', 'mf']

        X1 = df[features1].values
        X2 = df[features2].values
        X3 = df[features3].values
        X4 = df[features4].values
        data = {'0':X1, '1':X2, '2':X3, '3':X4, 'y':y}

    elif config.mod_split=='random':
        X = random_split(config, df.values)
        data = {'y':y}
        for i, x in enumerate(X):
            data[str(i)] = x

    elif config.mod_split=='computation_split':
        X = feature_split(config, df.values)
        data = {'y':y}
        for i, x in enumerate(X):
            data[str(i)] = x

    return data

def _msd(config):
    
    data_df = pd.read_csv(os.path.join(config.regression_datasets_dir, 'year_prediction.csv'))
    target_col = 'label'
    
    y = data_df[target_col]

    df = data_df.drop(columns=['label'])
    cols = df.columns.tolist()
    if config.mod_split=='none':
        X = df.values
        data = {'0':X, 'y':y}

    if config.mod_split=='human':
        features1 = cols[:12]
        features2 = cols[12:]

        X1 = df[features1].values
        X2 = df[features2].values
        data = {'0':X1, '1':X2, 'y':y}

    elif config.mod_split=='random':
        X = random_split(config, df.values)
        data = {'y':y}
        for i, x in enumerate(X):
            data[str(i)] = x

    elif config.mod_split=='computation_split':
        X = feature_split(config, df.values)
        data = {'y':y}
        for i, x in enumerate(X):
            data[str(i)] = x

    return data

def _life(config):
    data_df = pd.read_csv(os.path.join(config.regression_datasets_dir, 'life_expectancy.csv'))
    data_df[['Country']] = data_df[['Country']].apply(LabelEncoder().fit_transform)
    data_df[['Status']] = data_df[['Status']].apply(LabelEncoder().fit_transform)
    data_df = data_df.dropna()
    
    target_col = 'Life expectancy '
    
    y = np.asarray(data_df[target_col])

    df = data_df.drop(columns=[target_col])
    cols = df.columns.tolist()
    
    if config.mod_split=='none' or config.mod_split=='human':
        X = df.values
        data = {'0':X, 'y':y}

    elif config.mod_split=='random':
        X = random_split(config, df.values)
        data = {'y':y}
        for i, x in enumerate(X):
            data[str(i)] = x

    elif config.mod_split=='computation_split':
        X = feature_split(config, df.values)
        data = {'y':y}
        for i, x in enumerate(X):
            data[str(i)] = x

    return data
