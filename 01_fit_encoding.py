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
import feature_spaces
import sklearn.decomposition
import joblib
import os
import encoding_utils
# import encoding_models
from config import data_dir
import config
from ridge_utils.ridge import bootstrap_ridge, gen_temporal_chunk_splits
from ridge_utils.utils import make_delayed
import imodelsx.cache_save_utils
import story_names
import qa_questions
import random
import time

# get path to current file
path_to_file = os.path.dirname(os.path.abspath(__file__))


def nancorr(x, y):
    mask = ~np.isnan(x) & ~np.isnan(y)
    return np.corrcoef(x[mask], y[mask])[0, 1]

# python 01_fit_encoding.py --use_test_setup 1 --feature_space bert-10
# python 01_fit_encoding.py --use_test_setup 1 --feature_space qa_embedder-10
# python 01_fit_encoding.py --use_test_setup 1 --feature_space qa_embedder-5
# python 01_fit_encoding.py --use_test_setup 1 --feature_space qa_embedder-5 --qa_embedding_model mistralai/Mixtral-8x7B-v0.1


def add_main_args(parser):
    """Caching uses the non-default values from argparse to name the saving directory.
    Changing the default arg an argument will break cache compatibility with previous runs.
    """
    parser.add_argument("--subject", type=str, default='UTS03',
                        choices=['UTS01', 'UTS02', 'UTS03'],
                        help='top3 concatenates responses for S01-S03, useful for feature selection')
    parser.add_argument("--feature_space", type=str,
                        # default='distil-bert-tr2',
                        default='qa_embedder-10',
                        # default='distil-bert-10',
                        # qa_embedder-10
                        # default='qa_embedder-10',
                        #
                        # choices=sorted(list(
                        # feature_spaces._FEATURE_VECTOR_FUNCTIONS.keys())),
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
    parser.add_argument("--feature_selection_alpha_index", type=int,
                        default=-1,
                        help='in range(0, 100) - larger is more regularization')
    parser.add_argument("--qa_embedding_model", type=str,
                        default='mistralai/Mistral-7B-Instruct-v0.2',
                        # default='ensemble1',
                        help='Model to use for QA embedding, if feature_space is qa_embedder',
                        )
    parser.add_argument("--qa_questions_version", type=str, default='v1',
                        help='Which set of QA questions to use, if feature_space is qa_embedder')
    parser.add_argument("--l1_ratio", type=float,
                        default=0.5, help='l1 ratio for elasticnet (ignored if encoding_model is not elasticnet)')
    parser.add_argument("--min_alpha", type=float,
                        default=-1, help='min alpha, useful for forcing sparse coefs in elasticnet. Note: if too large, we arent really doing CV at all.')
    parser.add_argument('--pc_components', type=int, default=-1,
                        help='number of principal components to use (-1 doesnt use PCA at all). Note, use_test_setup alters this to 100.')
    parser.add_argument('--pc_components_input', type=int, default=-1,
                        help='number of principal components to use to transform features (-1 doesnt use PCA at all)')

    parser.add_argument("--mlp_dim_hidden", type=int,
                        help="hidden dim for MLP", default=512)
    parser.add_argument('--num_stories', type=int, default=-1,
                        help='number of stories to use (-1 for all). Note: use_test_setup alters this. Pass 0 to load shared stories (used for shared feature selection).')

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
    parser.add_argument(
        '--seed_stories',
        type=int,
        default=1,
        help='seed for order that stories are processed in',
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
        args.pc_components = 100
        # 'onapproachtopluto']  # , 'onapproachtopluto']
        # random.shuffle(story_names_test)
    # special case where we load shared stories
    elif args.num_stories == 0:
        story_names_train = story_names.get_story_names(
            'shared', 'train')
        story_names_test = story_names.get_story_names(
            'shared', 'test')

    elif args.num_stories > 0:
        story_names_train = story_names.get_story_names(
            args.subject, 'train')[:args.num_stories]
        story_names_test = story_names.get_story_names(
            args.subject, 'test')
    else:
        story_names_train = story_names.get_story_names(args.subject, 'train')
        story_names_test = story_names.get_story_names(args.subject, 'test')

    rng = np.random.default_rng(args.seed_stories)
    rng.shuffle(story_names_train)
    return story_names_train, story_names_test


def get_features_full(args, qa_embedding_model, story_names, extract_only=False):
    '''
    Params
    ------
    extract_only: bool
        if True, just run feature extraction and return

    Returns
    -------
    features_delayed: np.ndarray
        n_time_points x (n_delays x n_features)
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
        features_downsampled = encoding_utils.trim_and_normalize_features(
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

    resp_test = encoding_utils.get_response(
        story_names_test, subject)
    resp_train = encoding_utils.get_response(
        story_names_train, subject)

    if args.pc_components <= 0:
        return resp_train, resp_test
    else:
        logging.info('pc transforming resps...')

        pca_filename = join(data_dir, 'fmri_resp_norms',
                            subject, 'resps_pca.pkl')
        pca = joblib.load(pca_filename)
        pca.components_ = pca.components_[
            :args.pc_components]  # (n_components, n_voxels)

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
    features_delayed_distill = get_features_full(
        args_distill, args_distill.qa_embedding_model, story_names)
    preds_distilled = features_delayed_distill @ model_params['weights_pc']
    return preds_distilled


# def get_model(args):
#     if args.encoding_model == 'mlp':
#         return NeuralNetRegressor(
#             encoding_models.MLP(
#                 dim_inputs=stim_train_delayed.shape[1],
#                 dim_hidden=args.mlp_dim_hidden,
#                 dim_outputs=resp_train.shape[1]
#             ),
#             max_epochs=3000,
#             lr=1e-5,
#             optimizer=torch.optim.Adam,
#             callbacks=[EarlyStopping(patience=30)],
#             iterator_train__shuffle=True,
#             # device='cuda',
#         )


def fit_regression(args, r, features_train_delayed, resp_train, features_test_delayed, resp_test):
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
            features_train_delayed, resp_train, features_test_delayed, resp_test, alphas, args.nboots, args.chunklen,
            args.nchunks, singcutoff=args.singcutoff, single_alpha=args.single_alpha)

        # Save regression results
        model_params_to_save = {
            weights_key: wt,
            'alphas_best': alphas_best,
            # 'valinds': valinds
        }

        # corrs_tune is (alphas, voxels, and bootstrap samples)
        # now reorder so it's (voxels, alphas, bootstrap samples)
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
            num_splits=args.nboots, num_examples=features_train_delayed.shape[0],
            chunk_len=args.chunklen, num_chunks=args.nchunks)
        logging.info('Running elasticnet...')
        lin = MultiTaskElasticNetCV(
            alphas=alphas, cv=splits, n_jobs=10, l1_ratio=args.l1_ratio)
        lin.fit(features_train_delayed, resp_train)
        preds = lin.predict(features_test_delayed)
        corrs_test = []
        for i in range(preds.shape[1]):
            # np.corrcoef(resp_test[:, i], preds[:, i])[0, 1])
            corrs_test.append(nancorr(resp_test[:, i], preds[:, i]))
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
            rf.fit(features_train_delayed, resp_train[:, i])
            preds = rf.predict(features_test_delayed)
            # corrs_test.append(np.corrcoef(resp_test[:, i], preds)[0, 1])
            corrs_test.append(nancorr(resp_test[:, i], preds[:, i]))
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
        # corrs.append(
        # np.corrcoef(preds_voxels[:, i], resp[:, i])[0, 1])
        corrs.append(nancorr(preds_voxels[:, i], resp[:, i]))
    corrs = np.array(corrs)
    corrs[np.isnan(corrs)] = 0
    return corrs


def add_summary_stats(r, verbose=True):
    for key in ['corrs_test', 'corrs_tune', 'corrs_tune_pc', 'corrs_test_pc']:
        if key in r:
            r[key + '_mean'] = np.nanmean(r[key])
            r[key + '_median'] = np.nanmedian(r[key])
            r[key + '_frac>0'] = np.nanmean(r[key] > 0)
            r[key + '_mean_top1_percentile'] = np.nanmean(
                np.sort(r[key])[-len(r[key]) // 100:])
            r[key + '_mean_top5_percentile'] = np.nanmean(
                np.sort(r[key])[-len(r[key]) // 20:])

            # add r2 stats
            r[key.replace('corrs', 'r2') +
              '_mean'] = np.nanmean(r[key] * np.abs(r[key]))
            r[key.replace('corrs', 'r2') +
              '_median'] = np.nanmedian(r[key] * np.abs(r[key]))

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
    assert not (args.num_stories == 0 and args.feature_selection_alpha_index <
                0), 'num_stories == 0 should only be used during feature selection!'

    # set up logging
    logger = logging.getLogger()
    logging.basicConfig(level=logging.INFO)

    # set up saving directory + check for cache
    already_cached, save_dir_unique = imodelsx.cache_save_utils.get_save_dir_unique(
        parser, parser_without_computational_args, args, args.save_dir
    )
    if args.use_cache and already_cached and not args.use_test_setup:
        print(f"cached version exists! Successfully skipping :)\n\n\n")
        exit(0)
    for k in sorted(vars(args)):
        print("\t" + k + " " + str(vars(args)[k]))
    logging.info(f"\n\n\tsaving to " + save_dir_unique + "\n")

    # set seed
    t0 = time.time()
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    r = defaultdict(list)
    r.update(vars(args))
    r["save_dir_unique"] = save_dir_unique

    # get data
    story_names_train, story_names_test = get_story_names(args)
    if args.use_extract_only and args.pc_components_input < 0:
        all_stories = story_names_train + story_names_test
        random.shuffle(all_stories)
        get_features_full(args, args.qa_embedding_model,
                          all_stories, extract_only=True)
    print('loading features...')
    stim_test_delayed = get_features_full(
        args, args.qa_embedding_model, story_names_test)
    stim_train_delayed = get_features_full(
        args, args.qa_embedding_model, story_names_train)
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

    print('loading resps...')
    if not args.num_stories == 0:  # 0 is a special case which loads shared stories
        if args.pc_components <= 0:
            resp_train, resp_test = get_resps_full(
                args, args.subject, story_names_train, story_names_test)
        else:
            resp_train, resp_test, pca, scaler_train, scaler_test = get_resps_full(
                args, args.subject, story_names_train, story_names_test)

        # overwrite resp_train with distill model predictions
        if args.distill_model_path is not None:
            resp_train = get_resp_distilled(args, story_names_train)

    # select features
    if args.feature_selection_alpha_index >= 0:
        logging.info('selecting sparse feats...')
        # remove delays from stim
        stim_train = stim_train_delayed[:,
                                        :stim_train_delayed.shape[1] // args.ndelays]

        # coefs is (n_targets, n_features, n_alphas)
        alpha_range = (0, -3, 20)  # original was 0, -3, 15
        cache_file = join(config.repo_dir, 'sparse_feats_all_subj',
                          args.qa_questions_version + '_' + args.qa_embedding_model + '_' + str(alpha_range) + '.joblib')
        if os.path.exists(cache_file):
            alphas_enet, coefs_enet = joblib.load(cache_file)
        else:
            # get special resps by concatenating across subjects
            resp_train_shared = get_resps_full(
                args, 'shared', story_names_train, story_names_test)
            alphas_enet, coefs_enet, _ = enet_path(
                stim_train,
                resp_train_shared,
                l1_ratio=0.9,
                alphas=np.logspace(*alpha_range),
                verbose=3,
                max_iter=5000,  # defaults to 1000
                random_state=args.seed,
            )
            os.makedirs(join(config.repo_dir, 'sparse_feats'), exist_ok=True)
            joblib.dump((alphas_enet, coefs_enet), cache_file)
            logging.info(
                f"Succesfully completed feature selection {(time.time() - t0)/60:0.1f} minutes")
            exit(0)

        # pick the coefs
        coef_enet = coefs_enet[:, :, args.feature_selection_alpha_index]
        coef_nonzero = np.any(np.abs(coef_enet) > 0, axis=0)
        r['alpha'] = alphas_enet[args.feature_selection_alpha_index]
        r['weights_enet'] = coef_enet
        r['weight_enet_mask'] = coef_nonzero
        r['weight_enet_mask_num_nonzero'] = coef_nonzero.sum()

        # mask stim_delayed based on nonzero coefs (need to repeat by args.ndelays)
        coef_nonzero_rep = np.tile(
            coef_nonzero.flatten(), args.ndelays).flatten()
        stim_train_delayed = stim_train_delayed[:, coef_nonzero_rep]
        stim_test_delayed = stim_test_delayed[:, coef_nonzero_rep]

    # fit model
    r, model_params_to_save = fit_regression(
        args, r, stim_train_delayed, resp_train, stim_test_delayed, resp_test)

    # evaluate per voxel
    if args.pc_components > 0:
        resp_test = encoding_utils.get_response(
            story_names_test, args.subject)
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
    print(
        f"Succesfully completed in {(time.time() - t0)/60:0.1f} minutes, saved to {save_dir_unique}")
