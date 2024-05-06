from collections import defaultdict
import os.path
import pandas as pd
from sklearn.linear_model import MultiTaskElasticNetCV, enet_path
from copy import deepcopy
import torch
from sklearn.preprocessing import StandardScaler
import random
import logging
from sklearn.ensemble import RandomForestRegressor
from os.path import join, dirname
import argparse
import numpy as np
import ridge_utils.features.feature_spaces as feature_spaces
import sklearn.decomposition
import joblib
import os
from ridge_utils.data import response_utils
from ridge_utils.config import data_dir
import ridge_utils.config as config
from ridge_utils.encoding.ridge import bootstrap_ridge, gen_temporal_chunk_splits
from ridge_utils.data.utils import make_delayed
import imodelsx.cache_save_utils
import ridge_utils.data.story_names as story_names
import ridge_utils.features.qa_questions as qa_questions
import random
import time
from ridge_utils.data.utils import zscore


def trim_and_normalize_features(downsampled_feat, trim, normalize=True):
    """Trim and normalize the downsampled stimulus for train and test stories.

    Params
    ------
    stories
            List of stimuli stories.
    downsampled_feat (dict)
            Downsampled feature vectors for all stories.
    trim
            Trim downsampled stimulus matrix.
    """
    feat = [downsampled_feat[s][5+trim:-trim] for s in downsampled_feat]
    if normalize:
        feat = [zscore(f) for f in feat]
    feat = np.vstack(feat)
    return feat


def get_features_full(args, qa_embedding_model, story_names, extract_only=False):
    '''
    Params
    - -----
    extract_only: bool
        if True, just run feature extraction and return

    Returns
    - ------
    features_delayed: np.ndarray
        n_time_points x(n_delays x n_features)
    '''
    if qa_embedding_model == 'ensemble1':
        features_delayed_list = []
        for qa_embedding_model in ['mistralai/Mistral-7B-Instruct-v0.2', 'meta-llama/Meta-Llama-3-8B-Instruct', 'meta-llama/Meta-Llama-3-8B-Instruct-fewshot']:
            features_delayed = get_features_full(
                args, qa_embedding_model, story_names)
            features_delayed_list.append(features_delayed)
        features_avg = np.mean(features_delayed_list, axis=0)
        features_avg = features_avg / np.std(features_avg, axis=0)

    # for qa versions, we extract features multiple times and concatenate them
    # this helps with caching
    if 'qa_embedder' in args.feature_space:
        kwargs_list = qa_questions.get_kwargs_list_for_version_str(
            args.qa_questions_version)
    else:
        kwargs_list = [{}]

    features_downsampled_list = []
    for kwargs in kwargs_list:
        features_downsampled_dict = feature_spaces.get_features(
            args.feature_space,
            allstories=story_names,
            qa_embedding_model=qa_embedding_model,
            # use_cache=False,
            **kwargs)
        # n_time_points x n_features
        features_downsampled = trim_and_normalize_features(
            features_downsampled_dict, args.trim, normalize=True
        )
        features_downsampled_list.append(deepcopy(features_downsampled))
    torch.cuda.empty_cache()
    if extract_only:
        return

    features_downsampled_list = np.hstack(features_downsampled_list)
    # print('unique', np.unique(features_downsampled_list, return_counts=True))
    features_delayed = make_delayed(features_downsampled_list,
                                    delays=range(1, args.ndelays+1))
    return features_delayed
