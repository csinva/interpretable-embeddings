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
import encoding_utils
import encoding_models
from feature_spaces import _FEATURE_VECTOR_FUNCTIONS, get_feature_space, repo_dir, em_data_dir, data_dir, results_dir
from ridge_utils.ridge import bootstrap_ridge
import imodelsx.cache_save_utils
import story_names
import random

# get path to current file
path_to_file = os.path.dirname(os.path.abspath(__file__))


# python 01_fit_encoding.py --use_test_setup 1 --feature_space bert-10
# python 01_fit_encoding.py --use_test_setup 1 --feature_space qa_embedder-10
# python 01_fit_encoding.py --use_test_setup 1 --feature_space qa_embedder-5
