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

def add_main_args(parser):
    """Caching uses the non-default values from argparse to name the saving directory.
    Changing the default arg an argument will break cache compatibility with previous runs.
    """
    parser.add_argument("--subject", type=str, default='UTS03')
    parser.add_argument("--feature_space", type=str, default='distil-bert-10',  # qa_embedder-5
                        choices=list(_FEATURE_VECTOR_FUNCTIONS.keys()))
    parser.add_argument("--encoding_model", type=str, default='ridge')
    parser.add_argument("--trim", type=int, default=5)
    parser.add_argument("--ndelays", type=int, default=4)
    parser.add_argument("--nboots", type=int, default=50)
    parser.add_argument("--chunklen", type=int, default=40)
    parser.add_argument("--nchunks", type=int, default=125)
    parser.add_argument("--singcutoff", type=float, default=1e-10)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--pc_components', type=int, default=-1)
    parser.add_argument("-single_alpha", action="store_true")
    parser.add_argument("--mlp_dim_hidden", type=int,
                        help="hidden dim for MLP", default=512)
    parser.add_argument('--save_dir', type=str,
                        default=os.path.join(path_to_file, 'results'))
    parser.add_argument('--use_test_setup', type=int, default=1,
                        help='For fast testing - train/test on single story with 2 nboots.')
    return parser


def add_computational_args(parser):
    """Arguments that only affect computation and not the results (shouldnt use when checking cache)"""
    parser.add_argument(
        "--use_cache",
        type=int,
        default=1,
        choices=[0, 1],
        help="whether to check for cache",
    )
    parser.add_argument(
        "--use_save_features",
        type=int,
        default=1,
        choices=[0, 1],
        help="whether to save the constructed features",
    )
    return parser


def get_data(args):
    # Story names
    if args.use_test_setup:
        # train_stories = ['sloth']
        args.nboots = 20
        # train_stories = ['sloth', 'itsabox',
        #  'odetostepfather', 'inamoment', 'hangtime']
        train_stories = story_names.get_story_names(args.subject, 'train')
        # train_stories = [
        # 'adollshouse', 'adventuresinsayingyes', 'afatherscover', 'againstthewind', 'alternateithicatom', 'avatar',
        # 'backsideofthestorm', 'becomingindian', 'beneaththemushroomcloud',
        # ]
        random.shuffle(train_stories)
        # train_stories = train_stories
        test_stories = ['sloth', 'fromboyhoodtofatherhood']
        # 'onapproachtopluto']  # , 'onapproachtopluto']

    else:
        train_stories = story_names.get_story_names(args.subject, 'train')
        test_stories = story_names.get_story_names(args.subject, 'test')

    # Features
    features_downsampled = get_feature_space(
        args.feature_space, train_stories + test_stories)
    normalize = True if args.pc_components <= 0 else False
    stim_train_delayed = encoding_utils.add_delays(
        train_stories, features_downsampled, args.trim, args.ndelays, normalize=normalize)
    print("stim_train_delayed.shape: ", stim_train_delayed.shape)
    stim_test_delayed = encoding_utils.add_delays(
        test_stories, features_downsampled, args.trim, args.ndelays, normalize=normalize)
    print("stim_test_delayed.shape: ", stim_test_delayed.shape)

    # Response
    resp_train = encoding_utils.get_response(train_stories, args.subject)
    # (n_time_points x n_voxels), e.g. (9461, 95556)
    print("resp_train.shape", resp_train.shape)
    resp_test = encoding_utils.get_response(test_stories, args.subject)
    # (n_time_points x n_voxels), e.g. (291, 95556)
    print("resp_test.shape: ", resp_test.shape)

    # convert to pc components for predicting
    if args.pc_components > 0:
        pca_dir = join(data_dir, 'fmri_resp_norms', args.subject)
        pca = pkl.load(open(join(pca_dir, 'resps_pca.pkl'), 'rb'))['pca']
        pca.components_ = pca.components_[
            :args.pc_components]  # (n_components, n_voxels)
        # zRresp = zRresp @ comps.T
        resp_train = pca.transform(resp_train)  # [:, :args.pc_components]
        # resp_test_orig = deepcopy(resp_test)
        # zPresp = zPresp @ comps.T
        resp_test = pca.transform(resp_test)  # [:, :args.pc_components]
        print('reps_train.shape (after pca)', resp_train.shape)
        scaler_train = StandardScaler().fit(resp_train)
        scaler_test = StandardScaler().fit(resp_test)
        resp_train = scaler_train.transform(resp_train)
        resp_test = scaler_test.transform(resp_test)
        return stim_train_delayed, resp_train, stim_test_delayed, resp_test
    else:
        return stim_train_delayed, resp_train, stim_test_delayed, resp_test


def fit_regression(args, r, stim_train_delayed, resp_train, stim_test_delayed, resp_test):
    if args.pc_components > 0:
        alphas = np.logspace(1, 4, 10)
    else:
        alphas = np.logspace(1, 3, 10)

    if args.encoding_model == 'ridge':
        wt, corrs_test, alphas_best, corrs_tune, valinds = bootstrap_ridge(
            stim_train_delayed, resp_train, stim_test_delayed, resp_test, alphas, args.nboots, args.chunklen,
            args.nchunks, singcutoff=args.singcutoff, single_alpha=args.single_alpha)

        # Save regression results.
        model_params_to_save = {
            'weights': wt,
            'alphas_best': alphas_best,
            # 'valinds': valinds
        }
        # corrs_tune is (alphas, voxels, and bootstrap samples), so we average over the bootstrap samples and take the max over the alphas
        r['corrs_tune'] = corrs_tune.mean(axis=-1).max(axis=0)
        r['corrs_test'] = corrs_test
    elif args.encoding_model == 'mlp':
        stim_train_delayed = stim_train_delayed.astype(np.float32)
        resp_train = resp_train.astype(np.float32)
        stim_test_delayed = stim_test_delayed.astype(np.float32)
        net = NeuralNetRegressor(
            encoding_models.MLP(
                dim_inputs=stim_train_delayed.shape[1],
                dim_hidden=args.mlp_dim_hidden,
                dim_outputs=resp_train.shape[1]
            ),
            max_epochs=3000,
            lr=1e-5,
            optimizer=torch.optim.Adam,
            callbacks=[EarlyStopping(patience=30)],
            iterator_train__shuffle=True,
            # device='cuda',
        )
        net.fit(stim_train_delayed, resp_train)
        preds = net.predict(stim_test_delayed)
        corrs = []
        for i in range(preds.shape[1]):
            corrs.append(np.corrcoef(resp_test[:, i], preds[:, i])[0, 1])
        corrs = np.array(corrs)
        # print(corrs[:20])
        # c = corrs[~np.isnan(corrs)]
        # print('mean mlp corr', np.mean(c).round(3), 'max mlp corr',
        #       np.max(c).round(3), 'min mlp corr', np.min(c).round(3))
        r['corrs_test'] = corrs
        model_params_to_save = {
            'weights': net.module_.state_dict(),
        }

    # print('shapes', preds_voxels_test.shape, 'corrs', corrs.shape)
    # pca = pkl.load(open(join(pca_dir, 'resps_pca.pkl'), 'rb'))['pca']

        # np.savez("%s/corrs" % save_dir, corrs)
        # torch.save(net.module_.state_dict(), join(save_dir, 'weights.pt'))

    # save corrs for each voxel
    if args.pc_components > 0:
        r['corrs_test_pc'] = corrs

        """
        def _evaluate_pc_model_on_each_voxel(args, r, model_params_to_save):
            '''Todo: properly pass args here
            '''
            # np.savez("%s/corrs_pcs" % save_dir, corrs)
            if args.encoding_model == 'ridge':
                preds_pc_test = stim_test_delayed @ wt
            elif args.encoding_model == 'mlp':
                preds_pc_test = net.predict(stim_test_delayed)
            preds_voxels_test = pca.inverse_transform(
                scaler_test.inverse_transform(preds_pc_test)
            )  # (n_trs x n_voxels)
            # zPresp_orig (n_trs x n_voxels)
            # corrs: correlation list (n_voxels)
            # subtract mean over time points
            corrs = []
            for i in range(preds_voxels_test.shape[1]):
                corrs.append(
                    np.corrcoef(preds_voxels_test[:, i], resp_test_orig[:, i])[0, 1])
            corrs = np.array(corrs)
            # preds_normed = (preds_voxels_test - preds_voxels_test.mean(axis=0)) / reds_voxels_test
            # resps_normed = zPresp_orig - zPresp_orig.mean(axis=0)
            # corrs = np.diagonal(preds_normed.T @ resps_normed)

            return corrs
        
        r['corrs'] = _evaluate_pc_model_on_each_voxel(
            args, r, model_params_to_save)
        """
    return r, model_params_to_save


def add_summary_stats(r, verbose=True):
    for key in ['corrs_test', 'corrs_tune']:
        if key in r:
            r[key + '_mean'] = np.mean(r[key])
            r[key + '_median'] = np.median(r[key])
            r[key + '_frac>0'] = np.mean(r[key] > 0)
            r[key + '_mean_top1_percentile'] = np.mean(
                np.sort(r[key])[-len(r[key]) // 100:])
            r[key + '_mean_top5_percentile'] = np.mean(
                np.sort(r[key])[-len(r[key]) // 20:])

            if key == 'corrs_test' and verbose:
                logging.info(f"mean {key}: {r[key + '_mean']:.4f}")
                logging.info(f"median {key}: {r[key + '_median']:.4f}")
                logging.info(f"frac>0 {key}: {r[key + '_frac>0']:.4f}")
                logging.info(
                    f"mean top1 percentile {key}: {r[key + '_mean_top1_percentile']:.4f}")
                logging.info(
                    f"mean top5 percentile {key}: {r[key + '_mean_top5_percentile']:.4f}")
    return r


if __name__ == "__main__":
    # get args
    parser = argparse.ArgumentParser()
    parser_without_computational_args = add_main_args(parser)
    parser = add_computational_args(
        deepcopy(parser_without_computational_args))
    args = parser.parse_args()

    # set up logging
    logger = logging.getLogger()
    logging.basicConfig(level=logging.INFO)

    # parser = argparse.ArgumentParser()
    # args = parser.parse_args()
    # globals().update(args.__dict__)

    # set up saving directory + check for cache
    already_cached, save_dir_unique = imodelsx.cache_save_utils.get_save_dir_unique(
        parser, parser_without_computational_args, args, args.save_dir
    )

    if args.use_cache and already_cached:
        logging.info(f"cached version exists! Successfully skipping :)\n\n\n")
        exit(0)
    for k in sorted(vars(args)):
        logger.info("\t" + k + " " + str(vars(args)[k]))
    logging.info(f"\n\n\tsaving to " + save_dir_unique + "\n")

    # set seed
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    r = defaultdict(list)
    r.update(vars(args))
    r["save_dir_unique"] = save_dir_unique

    stim_train_delayed, resp_train, stim_test_delayed, resp_test = get_data(
        args)
    r, model_params_to_save = fit_regression(args, r, stim_train_delayed,
                                             resp_train, stim_test_delayed, resp_test)

    os.makedirs(save_dir_unique, exist_ok=True)
    r = add_summary_stats(r, verbose=True)
    joblib.dump(r, join(save_dir_unique, "results.pkl"))
    joblib.dump(model_params_to_save, join(
        save_dir_unique, "model_params.pkl"))
    logging.info(
        f"Succesfully completed, saved to {save_dir_unique}")
