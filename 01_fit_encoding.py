import os
import sys
import numpy as np
import h5py
import argparse
import json
import pathlib
from os.path import join, dirname
import logging
import random
from sklearn.preprocessing import StandardScaler
import torch
import pickle as pkl
from copy import deepcopy

# from .encoding_utils import *
import encoding_utils
from feature_spaces import _FEATURE_VECTOR_FUNCTIONS, get_feature_space, repo_dir, em_data_dir, data_dir, results_dir
from ridge_utils.ridge import bootstrap_ridge


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=str, default='UTS03')
    parser.add_argument("--feature", type=str, default='bert-10',
                        choices=list(_FEATURE_VECTOR_FUNCTIONS.keys()))
    parser.add_argument("--sessions", nargs='+',
                        type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--trim", type=int, default=5)
    parser.add_argument("--ndelays", type=int, default=4)
    parser.add_argument("--nboots", type=int, default=50)
    parser.add_argument("--chunklen", type=int, default=40)
    parser.add_argument("--nchunks", type=int, default=125)
    parser.add_argument("--singcutoff", type=float, default=1e-10)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--pc_components', type=int, default=-1)
    parser.add_argument("-use_corr", action="store_true")
    parser.add_argument("-single_alpha", action="store_true")
    parser.add_argument("--use_cache", type=int, default=1)

    # for faster testing
    parser.add_argument('--save_dir', type=str, default=None)
    parser.add_argument('-story_override', action='store_true',
                        help='For fast testing -- whether to train/test only on the sloth story.')
    logging.basicConfig(level=logging.INFO)

    args = parser.parse_args()
    globals().update(args.__dict__)
    print('args', vars(args))

    # set up saving....
    def get_save_dir(results_dir, feature, subject, ndelays, pc_components):
        if pc_components > 0:
            save_dir = join(results_dir, 'encoding', feature +
                            f'__ndel={ndelays}__pc={pc_components}', subject)
        else:
            save_dir = join(results_dir, 'encoding', feature +
                            f'__ndel={ndelays}', subject)
        return save_dir
    if args.save_dir is not None:
        save_dir = args.save_dir
    else:
        save_dir = get_save_dir(results_dir, args.feature,
                                args.subject, args.ndelays, args.pc_components)

    print("Saving encoding model & results to:", save_dir)
    if os.path.exists(join(save_dir, 'valinds.npz')) and args.use_cache:
        print('Already ran! Skipping....')
        exit(0)
    os.makedirs(save_dir, exist_ok=True)

    # Story names
    if args.story_override:
        train_stories = ['sloth']
        test_stories = ['sloth']
        allstories = ['sloth']
    else:
        assert np.amax(args.sessions) <= 5 and np.amin(
            args.sessions) >= 1, "1 <= session <= 5"
        train_stories, test_stories, allstories = encoding_utils.get_allstories(
            args.sessions)

    # Features
    downsampled_feat = get_feature_space(args.feature, allstories)
    print("Stimulus & Response parameters:")
    print("trim: %d, ndelays: %d" % (args.trim, args.ndelays))

    # Delayed stimulus
    normalize = True if args.pc_components <= 0 else False
    delRstim = encoding_utils.add_delays(
        train_stories, downsampled_feat, args.trim, args.ndelays, normalize=normalize)
    print("delRstim: ", delRstim.shape)
    delPstim = encoding_utils.add_delays(
        test_stories, downsampled_feat, args.trim, args.ndelays, normalize=normalize)
    print("delPstim: ", delPstim.shape)

    # Response
    zRresp = encoding_utils.get_response(train_stories, args.subject)
    # (n_time_points x n_voxels), e.g. (9461, 95556)
    print("zRresp: ", zRresp.shape)
    zPresp = encoding_utils.get_response(test_stories, args.subject)
    # (n_time_points x n_voxels), e.g. (291, 95556)
    print("zPresp: ", zPresp.shape)

    # convert to pc components for predicting
    if args.pc_components > 0:
        pca_dir = join(data_dir, 'fmri_resp_norms', args.subject)
        pca = pkl.load(open(join(pca_dir, 'resps_pca.pkl'), 'rb'))['pca']
        pca.components_ = pca.components_[
            :args.pc_components]  # (n_components, n_voxels)
        # zRresp = zRresp @ comps.T
        zRresp = pca.transform(zRresp)  # [:, :args.pc_components]
        zPresp_orig = deepcopy(zPresp)
        # zPresp = zPresp @ comps.T
        zPresp = pca.transform(zPresp)  # [:, :args.pc_components]
        print('ZRresp (after pca)', zRresp.shape)
        scaler_train = StandardScaler().fit(zRresp)
        scaler_test = StandardScaler().fit(zPresp)
        zRresp = scaler_train.transform(zRresp)
        zPresp = scaler_test.transform(zPresp)

    # Ridge
    if args.pc_components > 0:
        alphas = np.logspace(1, 4, 10)
    else:
        alphas = np.logspace(1, 3, 10)

    # Seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    print("Ridge parameters:")
    print("nboots: %d, chunklen: %d, nchunks: %d, single_alpha: %s, use_corr: %s" % (
        args.nboots, args.chunklen, args.nchunks, args.single_alpha, args.use_corr))

    wt, corrs, valphas, bscorrs, valinds = bootstrap_ridge(
        delRstim, zRresp, delPstim, zPresp, alphas, args.nboots, args.chunklen,
        args.nchunks, singcutoff=args.singcutoff, single_alpha=args.single_alpha,
        use_corr=args.use_corr)

    # Save regression results.
    np.savez("%s/weights" % save_dir, wt)
    np.savez("%s/corrs" % save_dir, corrs)
    np.savez("%s/valphas" % save_dir, valphas)
    np.savez("%s/bscorrs" % save_dir, bscorrs)
    np.savez("%s/valinds" % save_dir, np.array(valinds))
    print("Total r2: %d" % sum(corrs * np.abs(corrs)))

    if args.pc_components > 0:
        preds_voxels_test = pca.inverse_transform(
            scaler_test.inverse_transform(
                delRstim @ wt
            )
        )  # (n_trs x n_voxels)
        # zPresp_orig (n_trs x n_voxels)
        # corrs: correlation list (n_voxels)
        # subtract mean over time points
        corrs = []
        for i in range(preds_voxels_test.shape[1]):
            corrs.append(
                np.corrcoef(preds_voxels_test[:, i], zPresp_orig[:, i])[0, 1])
        corrs = np.array(corrs)
        # preds_normed = (preds_voxels_test - preds_voxels_test.mean(axis=0)) / reds_voxels_test
        # resps_normed = zPresp_orig - zPresp_orig.mean(axis=0)
        # corrs = np.diagonal(preds_normed.T @ resps_normed)
        print('mean voxel corr', np.mean(corrs).round(3), 'max voxel corr',
              np.max(corrs).round(3), 'min voxel corr', np.min(corrs).round(3))
        pkl.dump({
            'corrs': corrs, 'scaler': scaler_train},
            open(join(pca_dir, 'pc_results.pkl'), 'wb')
        )
        # print('shapes', preds_voxels_test.shape, 'corrs', corrs.shape)
        # pca = pkl.load(open(join(pca_dir, 'resps_pca.pkl'), 'rb'))['pca']
