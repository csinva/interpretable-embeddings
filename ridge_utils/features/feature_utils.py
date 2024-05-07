from copy import deepcopy
import torch
import numpy as np
import ridge_utils.features.feature_spaces as feature_spaces
import os
import ridge_utils.features.qa_questions as qa_questions
from ridge_utils.data.npp import zscore


def trim_and_normalize_features(downsampled_feat, trim=5, normalize=True):
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


def make_delayed(stim, delays, circpad=False):
    """Creates non-interpolated concatenated delayed versions of [stim] with the given [delays] 
    (in samples).

    If [circpad], instead of being padded with zeros, [stim] will be circularly shifted.
    """
    nt, ndim = stim.shape
    dstims = []
    for di, d in enumerate(delays):
        dstim = np.zeros((nt, ndim))
        if d < 0:  # negative delay
            dstim[:d, :] = stim[-d:, :]
            if circpad:
                dstim[d:, :] = stim[:-d, :]
        elif d > 0:
            dstim[d:, :] = stim[:-d, :]
            if circpad:
                dstim[:d, :] = stim[-d:, :]
        else:  # d==0
            dstim = stim.copy()
        dstims.append(dstim)
    return np.hstack(dstims)

# def add_delays(stim, ndelays):
#     """Get delayed stimulus matrix.
#     The stimulus matrix is delayed (typically by 2, 4, 6, 8 secs) to estimate the
#     hemodynamic response function with a Finite Impulse Response model.
#     """
#     # List of delays for Finite Impulse Response (FIR) model.
#     delays = range(1, ndelays+1)
#     delstim = make_delayed(stim, delays)
#     return delstim


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
    # for ensemble, recursively call this function and average the features
    if qa_embedding_model == 'ensemble1':
        features_delayed_list = []
        for qa_embedding_model in ['mistralai/Mistral-7B-Instruct-v0.2', 'meta-llama/Meta-Llama-3-8B-Instruct', 'meta-llama/Meta-Llama-3-8B-Instruct-fewshot']:
            features_delayed = get_features_full(
                args, qa_embedding_model, story_names)
            features_delayed_list.append(features_delayed)
        features_delayed_avg = np.mean(features_delayed_list, axis=0)
        # features_delayed_avg = features_delayed_avg / \
        # np.std(features_delayed_avg, axis=0)
        return features_delayed_avg

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
            features_downsampled_dict, normalize=True
        )
        features_downsampled_list.append(deepcopy(features_downsampled))
    torch.cuda.empty_cache()
    if extract_only:
        return

    features_downsampled_list = np.hstack(features_downsampled_list)
    features_delayed = make_delayed(features_downsampled_list,
                                    delays=range(1, args.ndelays+1))
    return features_delayed
