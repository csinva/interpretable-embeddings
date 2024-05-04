from collections import defaultdict
import os.path
from skorch import NeuralNetRegressor
from skorch.callbacks import EarlyStopping
from copy import deepcopy
import pickle as pkl
import torch
from sklearn.preprocessing import StandardScaler
import random
import logging
from os.path import join, dirname
import json
import argparse
import h5py
import numpy as np
import sys
import joblib
import os
import ridge_utils.encoding_utils as encoding_utils
from ridge_utils.ridge import bootstrap_ridge, gen_temporal_chunk_splits
import imodelsx.cache_save_utils
from sklearn.linear_model import MultiTaskElasticNetCV
import ridge_utils.data.story_names as story_names
import itertools as itools
import random

# get path to current file
path_to_file = os.path.dirname(os.path.abspath(__file__))


# python 01_fit_encoding.py --use_test_setup 1 --feature_space bert-10
# python 01_fit_encoding.py --use_test_setup 1 --feature_space qa_embedder-10
# python 01_fit_encoding.py --use_test_setup 1 --feature_space qa_embedder-5

if __name__ == '__main__':

    # generate synthetic test for bootstrap ridge
    np.random.seed(42)
    random.seed(42)

    X = np.random.randn(150, 20)
    W = np.random.randn(20, 10)
    Y = X @ W + np.random.randn(150, 10) / 10

    X_train, X_test = X[:100], X[100:]
    Y_train, Y_test = Y[:100], Y[100:]

    alphas = np.logspace(0, 3, 20)
    nboots = 3
    chunklen = 10
    nchunks = 5
    singcutoff = 1e-10
    single_alpha = False

    wt, corrs, alphas_best, corrs_tune, valinds = bootstrap_ridge(
        X_train, Y_train, X_test, Y_test, alphas, nboots, chunklen, nchunks, singcutoff=singcutoff, single_alpha=single_alpha)
    print('test corr', np.mean(corrs))

    splits = gen_temporal_chunk_splits(
        nboots, X_train.shape[0], chunklen, nchunks)
    m = MultiTaskElasticNetCV(alphas=alphas, cv=splits, n_jobs=1)
    m.fit(X_train, Y_train)
    preds = m.predict(X_test)
    print('test corr', np.mean([np.corrcoef(y, p)[0, 1]
          for y, p in zip(Y_test.T, preds.T)]))
    # print('test corr', m.score(X_test, Y_test))
