from pathlib import Path
import contextlib
import re
import wave
import six
from mutagen.mp3 import MP3

import numpy as np
import tensorflow as tf
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def normalize_compare_features(compare_train, compare_val, compare_features_size=21):
    sc = StandardScaler()
    sc.fit(compare_train)
    compare_train = sc.transform(compare_train)
    compare_val = sc.transform(compare_val)
    pca = PCA(n_components=compare_features_size)
    pca.fit(compare_train)
    compare_train = pca.transform(compare_train)
    compare_val = pca.transform(compare_val)

    return compare_train, compare_val