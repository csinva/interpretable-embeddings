import numpy as np
import time
import pathlib
import os
import h5py
from multiprocessing.pool import ThreadPool
from os.path import join, dirname
import json
from typing import List

from ridge_utils.npp import zscore, mcorr
from ridge_utils.utils import make_delayed
from feature_spaces import repo_dir, data_dir, em_data_dir


def get_allstories(sessions=[1, 2, 3, 4, 5]) -> List[str]:
    sessions = list(map(str, sessions))
    with open(join(em_data_dir, "sess_to_story.json"), "r") as f:
        sess_to_story = json.load(f)
    train_stories, test_stories = [], []
    for sess in sessions:
        stories, tstory = sess_to_story[sess][0], sess_to_story[sess][1]
        train_stories.extend(stories)
        if tstory not in test_stories:
            test_stories.append(tstory)
    assert len(set(train_stories) & set(test_stories)
               ) == 0, "Train - Test overlap!"
    allstories = list(set(train_stories) | set(test_stories))
    return train_stories, test_stories, allstories


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


def add_delays(stim, ndelays):
    """Get delayed stimulus matrix.
    The stimulus matrix is delayed (typically by 2, 4, 6, 8 secs) to estimate the
    hemodynamic response function with a Finite Impulse Response model.
    """
    # List of delays for Finite Impulse Response (FIR) model.
    delays = range(1, ndelays+1)
    delstim = make_delayed(stim, delays)
    return delstim


def get_response(stories, subject):
    """Get the subject"s fMRI response for stories."""
    main_path = pathlib.Path(__file__).parent.parent.resolve()
    subject_dir = join(
        data_dir, "ds003020/derivative/preprocessed_data/%s" % subject)
    base = os.path.join(main_path, subject_dir)
    resp = []
    for story in stories:
        resp_path = os.path.join(base, "%s.hf5" % story)
        hf = h5py.File(resp_path, "r")
        resp.extend(hf["data"][:])
        hf.close()
    return np.array(resp)


def get_permuted_corrs(true, pred, blocklen):
    nblocks = int(true.shape[0] / blocklen)
    true = true[:blocklen*nblocks]
    block_index = np.random.choice(range(nblocks), nblocks)
    index = []
    for i in block_index:
        start, end = i*blocklen, (i+1)*blocklen
        index.extend(range(start, end))
    pred_perm = pred[index]
    nvox = true.shape[1]
    corrs = np.nan_to_num(mcorr(true, pred_perm))
    return corrs


def permutation_test(true, pred, blocklen, nperms):
    start_time = time.time()
    pool = ThreadPool(processes=10)
    perm_rsqs = pool.map(
        lambda perm: get_permuted_corrs(true, pred, blocklen), range(nperms))
    pool.close()
    end_time = time.time()
    print((end_time - start_time) / 60)
    perm_rsqs = np.array(perm_rsqs).astype(np.float32)
    real_rsqs = np.nan_to_num(mcorr(true, pred))
    pvals = (real_rsqs <= perm_rsqs).mean(0)
    return np.array(pvals), perm_rsqs, real_rsqs
