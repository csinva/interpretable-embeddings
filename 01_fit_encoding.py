from collections import defaultdict
import os.path
import pandas as pd
from sklearn.linear_model import MultiTaskElasticNetCV
from skorch import NeuralNetRegressor
from skorch.callbacks import EarlyStopping
from copy import deepcopy
import pickle as pkl
import torch
from sklearn.preprocessing import StandardScaler
import random
import logging
from sklearn.ensemble import RandomForestRegressor
from os.path import join, dirname
import json
import argparse
import h5py
import numpy as np
import sys
import feature_spaces
import sklearn.decomposition
import joblib
import os
import encoding_utils
import encoding_models
from config import data_dir
from ridge_utils.ridge import bootstrap_ridge, gen_temporal_chunk_splits
from ridge_utils.utils import make_delayed
import imodelsx.cache_save_utils
import story_names
import qa_questions
import random

# get path to current file
path_to_file = os.path.dirname(os.path.abspath(__file__))


# python 01_fit_encoding.py --use_test_setup 1 --feature_space bert-10
# python 01_fit_encoding.py --use_test_setup 1 --feature_space qa_embedder-10
# python 01_fit_encoding.py --use_test_setup 1 --feature_space qa_embedder-5
# python 01_fit_encoding.py --use_test_setup 1 --feature_space qa_embedder-5 --qa_embedding_model mistralai/Mixtral-8x7B-v0.1

def add_main_args(parser):
    """Caching uses the non-default values from argparse to name the saving directory.
    Changing the default arg an argument will break cache compatibility with previous runs.
    """
    parser.add_argument("--subject", type=str, default='UTS03')
    parser.add_argument("--feature_space", type=str,
                        # default='distil-bert-tr2',
                        default='qa_embedder-10',
                        # default='distil-bert-10',
                        # qa_embedder-10
                        # default='qa_embedder-10',
                        #
                        choices=sorted(list(
                            feature_spaces._FEATURE_VECTOR_FUNCTIONS.keys())),
                        help='''Overloaded this argument.
                        qa_embedder-10 will run with ngram_context of 10 ngrams
                        qa_embedder-tr2 will run with tr_context of 2 TRs
                        qa_embedder-sec4 will run with ngram_context of 4 secs leading up to each word
                        '''
                        )
    parser.add_argument("--distill_model_path", type=str,
                        default=None,
                        # default='/home/chansingh/mntv1/deep-fMRI/encoding/results_apr7/68936a10a548e2b4ce895d14047ac49e7a56c3217e50365134f78f990036c5f7',
                        help='Path to saved pickles for distillation. Instead of fitting responses, fit the predictions of this model.')
    parser.add_argument("--encoding_model", type=str,
                        default='ridge',
                        # default='randomforest'
                        )
    parser.add_argument("--qa_embedding_model", type=str,
                        default='mistralai/Mistral-7B-Instruct-v0.2',
                        help='Model to use for QA embedding, if feature_space is qa_embedder',
                        choices=['mistralai/Mistral-7B-Instruct-v0.2',
                                 'mistralai/Mixtral-8x7B-Instruct-v0.1'],
                        )
    parser.add_argument("--qa_questions_version", type=str, default='v1',
                        help='Which set of QA questions to use, if feature_space is qa_embedder')
    parser.add_argument("--l1_ratio", type=float,
                        default=0.5, help='l1 ratio for elasticnet (ignored if encoding_model is not elasticnet)')
    parser.add_argument("--min_alpha", type=float,
                        default=-1, help='min alpha, useful for forcing sparse coefs in elasticnet. Note: if too large, we arent really doing CV at all.')
    parser.add_argument('--pc_components', type=int, default=-1,
                        help='number of principal components to use (-1 doesnt use PCA at all)')
    parser.add_argument('--pc_components_input', type=int, default=-1,
                        help='number of principal components to use to transform features (-1 doesnt use PCA at all)')

    parser.add_argument("--mlp_dim_hidden", type=int,
                        help="hidden dim for MLP", default=512)
    parser.add_argument('--num_stories', type=int, default=-1,
                        help='number of stories to use (-1 for all)')

    # linear modeling splits
    parser.add_argument("--trim", type=int, default=5)
    parser.add_argument("--ndelays", type=int, default=4)
    parser.add_argument("--nboots", type=int, default=50)
    parser.add_argument("--chunklen", type=int, default=40)
    # try to get nchunks * chunklen to ~20% of training data
    parser.add_argument("--nchunks", type=int, default=125)
    parser.add_argument("--singcutoff", type=float, default=1e-10)
    parser.add_argument("-single_alpha", action="store_true")

    # basic params
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--use_test_setup', type=int, default=1,
                        help='For fast testing - train/test on single story with 2 nboots.')
    return parser


def add_computational_args(parser):
    """Arguments that only affect computation and not the results (shouldnt use when checking cache)"""
    parser.add_argument('--save_dir', type=str,
                        default=os.path.join(path_to_file, 'results'))
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
    parser.add_argument(
        "--use_extract_only",
        type=int,
        default=1,
        choices=[0, 1],
        help="whether to jointly extract train/test (speeds things up if running over many seeds)",
    )
    return parser


def get_story_names(args):
    # Story names
    if args.use_test_setup:
        # train_stories = ['sloth']
        args.nboots = 3
        args.use_extract_only = 0
        story_names_train = ['sloth', 'adollshouse']
        # story_names_train = ['sloth']
        # story_names_train = story_names.get_story_names(args.subject, 'train')
        # story_names_train = [
        # 'adollshouse', 'adventuresinsayingyes', 'afatherscover', 'againstthewind', 'alternateithicatom', 'avatar',
        # 'backsideofthestorm', 'becomingindian', 'beneaththemushroomcloud',
        # ]
        # test_stories = ['sloth', 'fromboyhoodtofatherhood']
        story_names_test = ['fromboyhoodtofatherhood']
        # 'onapproachtopluto']  # , 'onapproachtopluto']
        # random.shuffle(story_names_test)
    elif args.num_stories > 0:
        story_names_train = story_names.get_story_names(
            args.subject, 'train')[:args.num_stories]
        story_names_test = story_names.get_story_names(
            args.subject, 'test')
    else:
        story_names_train = story_names.get_story_names(args.subject, 'train')
        story_names_test = story_names.get_story_names(args.subject, 'test')
    random.shuffle(story_names_train)
    return story_names_train, story_names_test


def get_data(args, story_names, extract_only=False):
    '''
    Params
    ------
    extract_only: bool
        if True, just run feature extraction and return

    Returns
    -------
    stim_delayed: np.ndarray
        n_time_points x (n_delays x n_features)
    resp: np.ndarray
        n_time_points x n_voxels (e.g. (27449, 95556) or (550, 95556))
    '''

    # for qa versions, we extract features multiple times and concatenate them
    # this helps with caching
    if 'qa_embedder' in args.feature_space:
        kwargs_list = qa_questions.get_kwargs_list_for_version_str(
            args.qa_questions_version)
        # if '-' in args.qa_questions_version:
        #     version_num = int(args.qa_questions_version.split('-')[0][1:])
        #     suffix = '-' + args.qa_questions_version.split('-')[1]
        # else:
        #     version_num = int(args.qa_questions_version[1])
        #     suffix = ''
        # kwargs_list = [{'qa_questions_version': f'v{i + 1}{suffix}'}
        #                for i in range(version_num)]
    else:
        kwargs_list = [{}]

    features_downsampled_list = []
    for kwargs in kwargs_list:
        # Features
        features_downsampled_dict = feature_spaces.get_features(
            args.feature_space,
            allstories=story_names,
            qa_embedding_model=args.qa_embedding_model,
            # use_cache=False,
            **kwargs)
        # for story_name in story_names:
        # print('unique after get_features', story_name, np.unique(
        # features_downsampled_dict[story_name], return_counts=True))
        # n_time_points x n_features
        features_downsampled = encoding_utils.trim_and_normalize_features(
            features_downsampled_dict, args.trim, normalize=True
        )
        features_downsampled_list.append(deepcopy(features_downsampled))
    torch.cuda.empty_cache()
    if extract_only:
        return

    features_downsampled_list = np.hstack(features_downsampled_list)
    # print('unique', np.unique(features_downsampled_list, return_counts=True))

    # n_time_points x (n_delays x n_features)
    stim_delayed = make_delayed(features_downsampled_list,
                                delays=range(1, args.ndelays+1))
    # stim_delayed = encoding_utils.add_delays(
    # story_names, features_downsampled_dict, args.trim, args.ndelays, normalize=normalize)

    # Response
    # use cached responses
    # if story_names_train == story_names.get_story_names(args.subject, 'train'):
    # resp_train = joblib.load(
    # join('/home/chansingh/cache_fmri_resps', f'{args.subject}.pkl'))
    # else:
    print('loading resps...')
    resp = encoding_utils.get_response(
        story_names, args.subject)
    assert resp.shape[0] == stim_delayed.shape[0], 'Resps loading for all stories, make sure to align with stim'

    return stim_delayed, resp


def transform_resps(args, resp_train, resp_test):
    pca_filename = join(data_dir, 'fmri_resp_norms',
                        args.subject, 'resps_pca.pkl')
    pca = joblib.load(pca_filename)
    pca.components_ = pca.components_[
        :args.pc_components]  # (n_components, n_voxels)
    resp_train = pca.transform(resp_train)
    resp_test = pca.transform(resp_test)
    print('reps_train.shape (after pca)', resp_train.shape)
    scaler_train = StandardScaler().fit(resp_train)
    scaler_test = StandardScaler().fit(resp_test)
    resp_train = scaler_train.transform(resp_train)
    resp_test = scaler_test.transform(resp_test)
    return resp_train, resp_test, pca, scaler_train, scaler_test


def get_resp_distilled(args, story_names):
    print('loading distill model...')
    args_distill = pd.Series(joblib.load(
        join(args.distill_model_path, 'results.pkl')))
    for k in ['subject', 'pc_components']:
        assert args_distill[k] == vars(args)[k], f'{k} mismatch'
    assert args_distill.pc_components > 0, 'distill only supported for pc_components > 0'

    model_params = joblib.load(
        join(args.distill_model_path, 'model_params.pkl'))
    stim_delayed_distill, _ = get_data(
        args_distill, story_names)
    preds_distilled = stim_delayed_distill @ model_params['weights_pc']
    return preds_distilled


def get_model(args):
    if args.encoding_model == 'mlp':
        return NeuralNetRegressor(
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


def fit_regression(args, r, stim_train_delayed, resp_train, stim_test_delayed, resp_test):
    if args.pc_components > 0:
        if args.min_alpha > 0:
            alphas = np.logspace(np.log10(args.min_alpha), 4, 12)
        else:
            alphas = np.logspace(1, 4, 12)
        weights_key = 'weights_pc'
        corrs_key_test = 'corrs_test_pc'
        corrs_key_tune = 'corrs_tune_pc'
    else:
        if args.min_alpha > 0:
            alphas = np.logspace(np.log10(args.min_alpha), 4, 12)
        else:
            alphas = np.logspace(1, 4, 12)
        weights_key = 'weights'
        corrs_key_test = 'corrs_test'
        corrs_key_tune = 'corrs_tune'

    if args.encoding_model == 'ridge':
        wt, corrs_test, alphas_best, corrs_tune, valinds = bootstrap_ridge(
            stim_train_delayed, resp_train, stim_test_delayed, resp_test, alphas, args.nboots, args.chunklen,
            args.nchunks, singcutoff=args.singcutoff, single_alpha=args.single_alpha)

        # Save regression results
        model_params_to_save = {
            weights_key: wt,
            'alphas_best': alphas_best,
            # 'valinds': valinds
        }

        # corrs_tune is (alphas, voxels, and bootstrap samples)

        # reorder be (voxels, alphas, bootstrap samples)
        corrs_tune = np.swapaxes(corrs_tune, 0, 1)
        # mean over bootstrap samples
        corrs_tune = corrs_tune.mean(axis=-1)

        # replace each element of alphas_best with its index in alphas
        alphas_idx = np.array([np.where(alphas == a)[0][0]
                              for a in alphas_best])

        # apply best alpha to each voxel
        corrs_tune = corrs_tune[np.arange(corrs_tune.shape[0]), alphas_idx]

        # so we average over the bootstrap samples and take the max over the alphas
        r[corrs_key_tune] = corrs_tune
        r[corrs_key_test] = corrs_test
    elif args.encoding_model == 'elasticnet':
        splits = gen_temporal_chunk_splits(
            num_splits=args.nboots, num_examples=stim_train_delayed.shape[0],
            chunk_len=args.chunklen, num_chunks=args.nchunks)
        print('Running elasticnet...')
        lin = MultiTaskElasticNetCV(
            alphas=alphas, cv=splits, n_jobs=10, l1_ratio=args.l1_ratio)
        lin.fit(stim_train_delayed, resp_train)
        preds = lin.predict(stim_test_delayed)
        corrs_test = []
        for i in range(preds.shape[1]):
            corrs_test.append(np.corrcoef(resp_test[:, i], preds[:, i])[0, 1])
        corrs_test = np.array(corrs_test)
        corrs_test[np.isnan(corrs_test)] = 0
        r[corrs_key_test] = corrs_test
        # r['mse_tune'] =
        model_params_to_save = {
            # want weights that are (n_features, n_targets)
            weights_key: lin.coef_.T,
            'alpha_best': lin.alpha_,
            'num_nonzero-coefs': np.sum(np.abs(lin.coef_) > 1e-8),
        }
    elif args.encoding_model == 'randomforest':
        rf = RandomForestRegressor(
            n_estimators=100, n_jobs=10)  # , max_depth=5)
        corrs_test = []
        for i in range(resp_train.shape[1]):
            rf.fit(stim_train_delayed, resp_train[:, i])
            preds = rf.predict(stim_test_delayed)
            corrs_test.append(np.corrcoef(resp_test[:, i], preds)[0, 1])
            print(i, 'rf corr', corrs_test[-1])
        corrs_test = np.array(corrs_test)
        corrs_test[np.isnan(corrs_test)] = 0
        r[corrs_key_test] = corrs_test
        model_params_to_save = {
            'weights': rf.feature_importances_,
        }

    # elif args.encoding_model == 'mlp':
    #     stim_train_delayed = stim_train_delayed.astype(np.float32)
    #     resp_train = resp_train.astype(np.float32)
    #     stim_test_delayed = stim_test_delayed.astype(np.float32)
    #     net = get_model(args)
    #     net.fit(stim_train_delayed, resp_train)
    #     preds = net.predict(stim_test_delayed)
    #     corrs_test = []
    #     for i in range(preds.shape[1]):
    #         corrs_test.append(np.corrcoef(resp_test[:, i], preds[:, i])[0, 1])
    #     corrs_test = np.array(corrs_test)
    #     r[corrs_key_test] = corrs_test
    #     model_params_to_save = {
    #         'weights': net.module_.state_dict(),
    #     }
        # torch.save(net.module_.state_dict(), join(save_dir, 'weights.pt'))
    return r, model_params_to_save


def evaluate_pc_model_on_each_voxel(
        args, stim, resp,
        model_params_to_save, pca, scaler):
    if args.encoding_model in ['ridge', 'elasticnet']:
        weights_pc = model_params_to_save['weights_pc']
        preds_pc = stim @ weights_pc
        model_params_to_save['weights'] = weights_pc * \
            scaler.scale_ @ pca.components_
        model_params_to_save['bias'] = scaler.mean_ @ pca.components_ + pca.mean_
        # note: prediction = stim @ weights + bias
    # elif args.encoding_model == 'mlp':
        # preds_pc_test = net.predict(stim_test_delayed)
    preds_voxels = pca.inverse_transform(
        scaler.inverse_transform(preds_pc)
    )  # (n_trs x n_voxels)
    corrs = []
    for i in range(preds_voxels.shape[1]):
        corrs.append(
            np.corrcoef(preds_voxels[:, i], resp[:, i])[0, 1])
    corrs = np.array(corrs)
    corrs[np.isnan(corrs)] = 0
    return corrs


def add_summary_stats(r, verbose=True):
    for key in ['corrs_test', 'corrs_tune', 'corrs_tune_pc', 'corrs_test_pc']:
        if key in r:
            r[key + '_mean'] = np.mean(r[key])
            r[key + '_median'] = np.median(r[key])
            r[key + '_frac>0'] = np.mean(r[key] > 0)
            r[key + '_mean_top1_percentile'] = np.mean(
                np.sort(r[key])[-len(r[key]) // 100:])
            r[key + '_mean_top5_percentile'] = np.mean(
                np.sort(r[key])[-len(r[key]) // 20:])

            # add r2 stats
            r[key.replace('corrs', 'r2') +
              '_mean'] = np.mean(r[key] * np.abs(r[key]))
            r[key.replace('corrs', 'r2') +
              '_median'] = np.median(r[key] * np.abs(r[key]))

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

    # set up saving directory + check for cache
    already_cached, save_dir_unique = imodelsx.cache_save_utils.get_save_dir_unique(
        parser, parser_without_computational_args, args, args.save_dir
    )

    if args.use_cache and already_cached and not args.use_test_setup:
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

    # get data
    story_names_train, story_names_test = get_story_names(args)
    if args.use_extract_only:
        all_stories = story_names_train + story_names_test
        random.shuffle(all_stories)
        get_data(args, all_stories, extract_only=True)
    stim_train_delayed, resp_train = get_data(args, story_names_train)
    print('stim_train.shape', stim_train_delayed.shape,
          'resp_train.shape', resp_train.shape)
    stim_test_delayed, resp_test = get_data(args, story_names_test)
    print('stim_test.shape', stim_test_delayed.shape,
          'resp_test.shape', resp_test.shape)
    if args.pc_components > 0:
        print('pc transforming resps...')
        resp_train, resp_test, pca, scaler_train, scaler_test = transform_resps(
            args, resp_train, resp_test)

    # overwrite resp_train with distill model predictions
    if args.distill_model_path is not None:
        resp_train = get_resp_distilled(args, story_names_train)

    # fit pca and project with less components
    if args.pc_components_input > 0:
        print('fitting pca to inputs...', args.pc_components_input, 'components')
        pca_input = sklearn.decomposition.PCA(
            n_components=args.pc_components_input)
        stim_train_delayed = pca_input.fit_transform(stim_train_delayed)
        stim_test_delayed = pca_input.transform(stim_test_delayed)
        scaler_input_train = StandardScaler().fit(stim_train_delayed)
        stim_train_delayed = scaler_input_train.transform(stim_train_delayed)
        scaler_input_test = StandardScaler().fit(stim_test_delayed)
        stim_test_delayed = scaler_input_test.transform(stim_test_delayed)

    # fit model
    r, model_params_to_save = fit_regression(
        args, r, stim_train_delayed, resp_train, stim_test_delayed, resp_test)

    # evaluate per voxel
    if args.pc_components > 0:
        stim_test_delayed, resp_test = get_data(args, story_names_test)
        if args.pc_components_input > 0:
            stim_test_delayed = pca_input.transform(stim_test_delayed)
            stim_test_delayed = scaler_input_test.transform(stim_test_delayed)
        r['corrs_test'] = evaluate_pc_model_on_each_voxel(
            args, stim_test_delayed, resp_test,
            model_params_to_save, pca, scaler_test)
        model_params_to_save['pca'] = pca
        model_params_to_save['scaler_test'] = scaler_test
        model_params_to_save['scaler_train'] = scaler_train

    # add extra stats
    r = add_summary_stats(r, verbose=True)

    os.makedirs(save_dir_unique, exist_ok=True)
    joblib.dump(r, join(save_dir_unique, "results.pkl"))
    joblib.dump(model_params_to_save, join(
        save_dir_unique, "model_params.pkl"))
    logging.info(
        f"Succesfully completed, saved to {save_dir_unique}")
