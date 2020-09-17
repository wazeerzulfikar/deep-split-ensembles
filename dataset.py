import os
import numpy as np
np.random.seed(0)

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from sklearn.preprocessing import LabelEncoder
from sklearn.datasets import load_boston
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import scale
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import pdist

import alzheimers.dataset as alzheimers_dataset

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
    # if config.dataset == 'energy_efficiency':
    #     linkage = np.load('energy_efficiency_linkage.npy')
    if config.dataset=='msd':
        clusters = fcluster(linkage, 0.75 * Y.max(), 'distance')
    else:
        clusters = fcluster(linkage, config.hc_threshold * Y.max(), 'distance')

    if(return_split_sizes):
        return clusters

    X = []
    for cluster in set(clusters):
        indices = [j for j in range(len(clusters)) if clusters[j]==cluster]
        # print(indices)
        X.append(np.transpose(data[indices]))
    return X

def feature_as_a_cluster(config, features):
    X = []
    for idx in range(features.shape[-1]):
        X.append(features[:,idx].reshape(-1,1))
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

    elif config.dataset=='alzheimers':
        data = _alzheimers(config)

    elif 'alzheimers_test' in config.dataset:
        data = _alzheimers_test(config)

    elif config.dataset == 'toy':
        x_limits = [-4, 4]

        n_datapoints = 40
        x1 = np.random.uniform(x_limits[0], x_limits[1], size=(n_datapoints,1))
        e1 = np.random.normal(loc=0, scale=3, size=(n_datapoints,1))
        x2 = np.random.uniform(x_limits[0], x_limits[1], size=(n_datapoints,1))
        e2 = np.random.normal(loc=1, scale=2, size=(n_datapoints,1))

        y = (np.power(x1, config.power) + e1) * (np.power(x2, config.power) + e2)


        data = {'0': x1, '1': x2, 'y': y} #Simulated later

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

    elif config.mod_split=='feature_as_a_cluster':
        X = feature_as_a_cluster(config, df.values)
        data = {'y':y}
        for i, x in enumerate(X):
            data[str(i)] = x

    return data

def _cement(config):
    data_df = pd.read_csv(os.path.join(config.datasets_dir, 'cement.csv'))
    
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

    elif config.mod_split=='feature_as_a_cluster':
        X = feature_as_a_cluster(config, df.values)
        data = {'y':y}
        for i, x in enumerate(X):
            data[str(i)] = x

    return data

def _energy_efficiency(config):

    data_df = pd.read_csv(os.path.join(config.datasets_dir, 'energy_efficiency.csv'))
    
    target_col1 = 'Heating Load'
    target_col2 = 'Cooling Load'
    y = data_df[target_col1]

    df = data_df.drop(columns=[target_col1, target_col2])

    if config.mod_split=='none':
        X = df.values
        data = {'0':X, 'y':y}

    if config.mod_split=='human':
        # X5
        # X1, X2, X3, X4
        # X7, X8
        # X6

        features1 = ['Overall Height']
        features2 = ['Relative Compactness', 'Surface Area', 'Wall Area', 'Roof Area']
        features3 = ['Glazing Area', 'Glazing Area Distribution']
        features4 = ['Orientation']

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

    elif config.mod_split=='feature_as_a_cluster':
        X = feature_as_a_cluster(config, df.values)
        data = {'y':y}
        for i, x in enumerate(X):
            data[str(i)] = x

    return data

def _kin8nm(config):
    data_df = pd.read_csv(os.path.join(config.datasets_dir, 'kin8nm.csv'))
    
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

    elif config.mod_split=='feature_as_a_cluster':
        X = feature_as_a_cluster(config, df.values)
        data = {'y':y}
        for i, x in enumerate(X):
            data[str(i)] = x

    return data

def _power_plant(config):
    data_df = pd.read_csv(os.path.join(config.datasets_dir, 'power_plant.csv'))
    
    y = data_df['PE']

    df = data_df.drop(columns=['PE'])
    if config.mod_split=='none':
        X = df.values
        data = {'0':X, 'y':y}

    elif config.mod_split=='human':
        # T, AP, RH
        # V

        features1 = ['AT', 'AP', 'RH']
        features2 = ['V']
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

    elif config.mod_split=='feature_as_a_cluster':
        X = feature_as_a_cluster(config, df.values)
        data = {'y':y}
        for i, x in enumerate(X):
            data[str(i)] = x

    return data

def _protein(config):
    data_df = pd.read_csv(os.path.join(config.datasets_dir, 'protein.csv'))
    
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

    elif config.mod_split=='feature_as_a_cluster':
        X = feature_as_a_cluster(config, df.values)
        data = {'y':y}
        for i, x in enumerate(X):
            data[str(i)] = x

    return data

def _wine(config):
    data_df = pd.read_csv(os.path.join(config.datasets_dir, 'wine.csv'))
    
    y = data_df['quality']

    df = data_df.drop(columns=['quality'])

    if config.mod_split=='none':
        X = df.values
        data = {'0':X, 'y':y}

    if config.mod_split=='human':
        features1 = ['alcohol', 'pH', 'fixed acidity', 'density', 'residual sugar']
        features2 = ['volatile acidity', 'citric acid']
        features3 = ['chlorides','free sulfur dioxide', 'total sulfur dioxide', 'sulphates']

        X1 = df[features1].values
        X2 = df[features2].values
        X3 = df[features3].values
        data = {'0':X1, '1':X2, '2':X3, 'y':y}

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

    elif config.mod_split=='feature_as_a_cluster':
        X = feature_as_a_cluster(config, df.values)
        data = {'y':y}
        for i, x in enumerate(X):
            data[str(i)] = x

    return data

def _yacht(config):
    cols = ['Longitudinal position of the center of buoyancy', 'Prismatic coefficient', 'Length-displacement ratio', 
            'Beam-draught ratio', 'Length-beam ratio', 'Froude number', 'Residuary resistance per unit weight of displacement']

    data_df = pd.read_csv(os.path.join(config.datasets_dir, 'yacht.data'), sep="\\s+", names=cols)
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

    elif config.mod_split=='feature_as_a_cluster':
        X = feature_as_a_cluster(config, df.values)
        data = {'y':y}
        for i, x in enumerate(X):
            data[str(i)] = x

    return data

def _naval(config):
    cols = ['lp', 'v', 'gtt', 'gtn', 'ggn', 'ts', 'tp',
            't48', 't1', 't2', 'p48', 'p1', 'p2', 'pexh',
            'tic', 'mf', 'y1', 'y2']

    data_df = pd.read_csv(os.path.join(config.datasets_dir, 'naval.csv'), sep='\\s+', names=cols)
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

    elif config.mod_split=='feature_as_a_cluster':
        X = feature_as_a_cluster(config, df.values)
        data = {'y':y}
        for i, x in enumerate(X):
            data[str(i)] = x

    return data

def _msd(config):
    
    data_df = pd.read_csv(os.path.join(config.datasets_dir, 'year_prediction.csv'))
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

    elif config.mod_split=='feature_as_a_cluster':
        X = feature_as_a_cluster(config, df.values)
        data = {'y':y}
        for i, x in enumerate(X):
            data[str(i)] = x

    return data

def _life(config):
    data_df = pd.read_csv(os.path.join(config.datasets_dir, 'life_expectancy.csv'))
    data_df[['Country']] = data_df[['Country']].apply(LabelEncoder().fit_transform)
    data_df[['Status']] = data_df[['Status']].apply(LabelEncoder().fit_transform)
    data_df = data_df.dropna()
    
    target_col = 'Life expectancy '

    y = data_df[target_col]

    df = data_df.drop(columns=[target_col])
    cols = df.columns.tolist()
    
    if config.mod_split=='none':
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


class EasyDict(dict):
    def __init__(self, *args, **kwargs): super().__init__(*args, **kwargs)
    def __getattr__(self, name): return self[name]
    def __setattr__(self, name, value): self[name] = value
    def __delattr__(self, name): del self[name] 

def _alzheimers(config):
    alzheimers_config = EasyDict({
        # 'task': 'classification',
        'task': 'regression',

        # 'dataset_dir': '../DementiaBank'
        'dataset_dir': 'datasets/ADReSS-IS2020-data/train',

        'longest_speaker_length': 32,
        'n_pause_features': 11,
        'compare_features_size': 21
    })
    alzheimers_data = alzheimers_dataset.prepare_data(alzheimers_config, select_gender=config.select_gender)

    data = {}
    data['y'] = alzheimers_data['y_reg']
    data['0'] = alzheimers_data['intervention']
    data['1'] = alzheimers_data['pause']
    data['2'] = alzheimers_data['compare']
    return data

def _alzheimers_test(config):
    test_data_name = config.dataset.split('_')[-1]

    if test_data_name in ['male', 'female']:
        print(test_data_name)
        config.select_gender = test_data_name
        return _alzheimers(config)

    alzheimers_config = EasyDict({
        'dataset_dir': 'datasets/ADReSS-IS2020-data/{}'.format(test_data_name),

        'longest_speaker_length': 32,
        'n_pause_features': 11,
        'compare_features_size': 21,
    })
    alzheimers_data = alzheimers_dataset.prepare_test_data(alzheimers_config)

    data = {}
    data['y'] = alzheimers_data['y_reg']
    data['0'] = alzheimers_data['intervention']
    data['1'] = alzheimers_data['pause']
    data['2'] = alzheimers_data['compare']

    min_len = min([len(data[i]) for i in data])
    data['0'] = data['0'][:min_len]
    data['1'] = data['1'][:min_len]
    data['2'] = data['2'][:min_len]
    data['y'] = data['y'][:min_len]

    return data