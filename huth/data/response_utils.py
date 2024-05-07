# from huth.data.npp import mcorr
# from typing import List
# import json
from os.path import join, dirname
# from multiprocessing.pool import ThreadPool
import h5py
import pathlib
# import time
import os.path
import pandas as pd
from sklearn.preprocessing import StandardScaler
from os.path import join
import logging
import numpy as np
import joblib
import os
import huth.features
import huth.config as config
# import random


def load_response(stories, subject):
    """Get the subject"s fMRI response for stories."""
    main_path = pathlib.Path(__file__).parent.parent.resolve()
    subject_dir = join(
        config.root_dir, 'data', f"ds003020/derivative/preprocessed_data/{subject}")
    base = os.path.join(main_path, subject_dir)
    resp = []
    for story in stories:
        resp_path = os.path.join(base, f"{story}.hf5")
        hf = h5py.File(resp_path, "r")
        resp.extend(hf["data"][:])
        hf.close()
    return np.array(resp)


def load_pca(subject, pc_components=None):
    if pc_components == 100:
        pca_filename = join(config.resp_processing_dir,
                            subject, 'resps_pca_100.pkl')
        return joblib.load(pca_filename)
    else:
        pca_filename = join(config.resp_processing_dir,
                            subject, 'resps_pca.pkl')
        pca = joblib.load(pca_filename)
        pca.components_ = pca.components_[
            :pc_components]
        return pca


def get_resps_full(
    args, subject, story_names_train, story_names_test
):
    '''
    resp_train: np.ndarray
        n_time_points x n_voxels
    '''
    if subject == 'shared':
        resp_train, _, _, _, _ = get_resps_full(
            args, 'UTS01', story_names_train, story_names_test)
        resp_train2, _, _, _, _ = get_resps_full(
            args, 'UTS02', story_names_train, story_names_test)
        resp_train3, _, _, _, _ = get_resps_full(
            args, 'UTS03', story_names_train, story_names_test)
        resp_train = np.hstack([resp_train, resp_train2, resp_train3])
        return resp_train

    resp_test = load_response(
        story_names_test, subject)
    resp_train = load_response(
        story_names_train, subject)

    if args.pc_components <= 0:
        return resp_train, resp_test
    else:
        logging.info('pc transforming resps...')

        # pca.components_ is (n_components, n_voxels)
        pca = load_pca(subject, args.pc_components)

        # fill nans with nanmean
        resp_train[np.isnan(resp_train)] = np.nanmean(resp_train)
        resp_test[np.isnan(resp_test)] = np.nanmean(resp_test)

        resp_train = pca.transform(resp_train)
        resp_test = pca.transform(resp_test)
        logging.info(f'reps_train.shape (after pca) {resp_train.shape}')
        scaler_train = StandardScaler().fit(resp_train)
        scaler_test = StandardScaler().fit(resp_test)
        resp_train = scaler_train.transform(resp_train)
        resp_test = scaler_test.transform(resp_test)
        return resp_train, resp_test, pca, scaler_train, scaler_test


def get_resp_distilled(args, story_names):
    logging.info('loading distill model...')
    args_distill = pd.Series(joblib.load(
        join(args.distill_model_path, 'results.pkl')))
    for k in ['subject', 'pc_components']:
        assert args_distill[k] == vars(args)[k], f'{k} mismatch'
    assert args_distill.pc_components > 0, 'distill only supported for pc_components > 0'

    model_params = joblib.load(
        join(args.distill_model_path, 'model_params.pkl'))
    features_delayed_distill = huth.features.get_features_full(
        args_distill, args_distill.qa_embedding_model, story_names)
    preds_distilled = features_delayed_distill @ model_params['weights_pc']
    return preds_distilled


# def get_permuted_corrs(true, pred, blocklen):
#     nblocks = int(true.shape[0] / blocklen)
#     true = true[:blocklen*nblocks]
#     block_index = np.random.choice(range(nblocks), nblocks)
#     index = []
#     for i in block_index:
#         start, end = i*blocklen, (i+1)*blocklen
#         index.extend(range(start, end))
#     pred_perm = pred[index]
#     nvox = true.shape[1]
#     corrs = np.nan_to_num(mcorr(true, pred_perm))
#     return corrs


# def permutation_test(true, pred, blocklen, nperms):
#     start_time = time.time()
#     pool = ThreadPool(processes=10)
#     perm_rsqs = pool.map(
#         lambda perm: get_permuted_corrs(true, pred, blocklen), range(nperms))
#     pool.close()
#     end_time = time.time()
#     print((end_time - start_time) / 60)
#     perm_rsqs = np.array(perm_rsqs).astype(np.float32)
#     real_rsqs = np.nan_to_num(mcorr(true, pred))
#     pvals = (real_rsqs <= perm_rsqs).mean(0)
#     return np.array(pvals), perm_rsqs, real_rsqs
