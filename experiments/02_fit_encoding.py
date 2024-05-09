from collections import defaultdict
import os.path
from sklearn.linear_model import enet_path
from copy import deepcopy
import torch
import random
import logging
from sklearn.ensemble import RandomForestRegressor
from os.path import join, dirname
import argparse
import numpy as np
import joblib
import os
from neuro1.data import response_utils
from neuro1.features import feature_utils
import neuro1.config as config
from neuro1.encoding.ridge import bootstrap_ridge, gen_temporal_chunk_splits
import imodelsx.cache_save_utils
import neuro1.data.story_names as story_names
import random
import time
from neuro1.encoding.eval import nancorr, evaluate_pc_model_on_each_voxel, add_summary_stats

# get path to current file
path_to_repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
path_to_file = os.path.dirname(os.path.abspath(__file__))


def add_main_args(parser):
    """Caching uses the non-default values from argparse to name the saving directory.
    Changing the default arg an argument will break cache compatibility with previous runs.
    """
    # data arguments
    parser.add_argument("--subject", type=str, default='UTS03',
                        choices=['UTS01', 'UTS02', 'UTS03'],
                        help='top3 concatenates responses for S01-S03, useful for feature selection')
    parser.add_argument('--pc_components', type=int, default=-1,
                        help='''number of principal components to use for reducing output(-1 doesnt use PCA at all).
                        Note, use_test_setup alters this to 100.''')
    parser.add_argument("--distill_model_path", type=str,
                        default=None,
                        # default='/home/chansingh/mntv1/deep-fMRI/encoding/results_apr7/68936a10a548e2b4ce895d14047ac49e7a56c3217e50365134f78f990036c5f7',
                        help='Path to saved pickles for distillation. Instead of fitting responses, fit the predictions of this model.')

    # encoding
    parser.add_argument("--feature_space", type=str,
                        default='qa_embedder-10',
                        # choices=['qa_embedder-10', 'finetune_roberta-base-10'],
                        help='''Overloaded this argument.
                        qa_embedder-10 will run with ngram_context of 10 ngrams
                        qa_embedder-tr2 will run with tr_context of 2 TRs
                        qa_embedder-sec4 will run with ngram_context of 4 secs leading up to each word
                        distil-bert-10 will extract embeddings from distil-bert
                        finetune_roberta-base-10 will run finetuned roberta model with 10 ngram_context
                        '''
                        )
    parser.add_argument('--num_stories', type=int, default=-1,
                        help='number of stories to use (-1 for all). Note: use_test_setup alters this. Pass 0 to load shared stories (used for shared feature selection).')
    parser.add_argument("--feature_selection_alpha_index", type=int,
                        default=-1,
                        help='in range(0, 100) - larger is more regularization')

    # qa features
    parser.add_argument("--qa_embedding_model", type=str,
                        default='mistralai/Mistral-7B-Instruct-v0.2',
                        # default='ensemble1',
                        help='Model to use for QA embedding, if feature_space is qa_embedder',
                        )
    parser.add_argument("--qa_questions_version", type=str,
                        default='v1',
                        # default='v3_boostexamples',
                        help='Which set of QA questions to use, if feature_space is qa_embedder')

    # linear modeling
    parser.add_argument("--encoding_model", type=str,
                        default='ridge',
                        # default='randomforest'
                        )

    parser.add_argument("--ndelays", type=int, default=4)
    parser.add_argument("--nboots", type=int, default=50)
    parser.add_argument("--chunklen", type=int, default=40,
                        help='try to get nchunks * chunklen to ~20% of training data')
    parser.add_argument("--nchunks", type=int, default=125)
    parser.add_argument("--singcutoff", type=float, default=1e-10)
    parser.add_argument("-single_alpha", action="store_true")
    # parser.add_argument("--trim", type=int, default=5) # always end up using 5
    # parser.add_argument("--l1_ratio", type=float,
    # default=0.5, help='l1 ratio for elasticnet (ignored if encoding_model is not elasticnet)')
    # parser.add_argument("--min_alpha", type=float,
    # default=-1, help='min alpha, useful for forcing sparse coefs in elasticnet. Note: if too large, we arent really doing CV at all.')
    # parser.add_argument('--pc_components_input', type=int, default=-1,
    # help='number of principal components to use to transform features (-1 doesnt use PCA at all)')
    # parser.add_argument("--mlp_dim_hidden", type=int,
    # help="hidden dim for MLP", default=512)

    # basic params
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--use_test_setup', type=int, default=1,
                        help='For fast testing - train/test on single story with 2 nboots.')
    return parser


def add_computational_args(parser):
    """Arguments that only affect computation and not the results (shouldnt use when checking cache)"""
    parser.add_argument('--save_dir', type=str,
                        default=os.path.join(path_to_repo, 'results'))
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


def fit_regression(args, r, features_train_delayed, resp_train, features_test_delayed, resp_test):
    if args.pc_components > 0:
        # if args.min_alpha > 0:
        # alphas = np.logspace(np.log10(args.min_alpha), 4, 12)
        # else:
        alphas = np.logspace(1, 4, 12)
        weights_key = 'weights_pc'
        corrs_key_test = 'corrs_test_pc'
        corrs_key_tune = 'corrs_tune_pc'
    else:
        # if args.min_alpha > 0:
        # alphas = np.logspace(np.log10(args.min_alpha), 4, 12)
        # else:
        alphas = np.logspace(1, 4, 12)
        weights_key = 'weights'
        corrs_key_test = 'corrs_test'
        corrs_key_tune = 'corrs_tune'

    if args.encoding_model == 'ridge':
        wt, corrs_test, alphas_best, corrs_tune, valinds = bootstrap_ridge(
            features_train_delayed, resp_train, features_test_delayed, resp_test,
            alphas, args.nboots, args.chunklen,
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

    return r, model_params_to_save


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
    if args.use_extract_only:
        all_stories = story_names_train + story_names_test
        random.shuffle(all_stories)
        feature_utils.get_features_full(args, args.qa_embedding_model,
                                        all_stories, extract_only=True)
    print('loading features...')
    stim_test_delayed = feature_utils.get_features_full(
        args, args.qa_embedding_model, story_names_test)
    stim_train_delayed = feature_utils.get_features_full(
        args, args.qa_embedding_model, story_names_train)

    print('loading resps...')
    if not args.num_stories == 0:  # 0 is a special case which loads shared stories
        if args.pc_components <= 0:
            resp_train, resp_test = response_utils.get_resps_full(
                args, args.subject, story_names_train, story_names_test)
        else:
            resp_train, resp_test, pca, scaler_train, scaler_test = response_utils.get_resps_full(
                args, args.subject, story_names_train, story_names_test)

        # overwrite resp_train with distill model predictions
        if args.distill_model_path is not None:
            resp_train = response_utils.get_resp_distilled(
                args, story_names_train)

    # select features
    if args.feature_selection_alpha_index >= 0:
        print('selecting sparse feats...')
        # remove delays from stim
        stim_train = stim_train_delayed[:,
                                        :stim_train_delayed.shape[1] // args.ndelays]

        # coefs is (n_targets, n_features, n_alphas)
        if args.num_stories == 0:
            cache_dir = join(config.root_dir, 'qa', 'sparse_feats_all_subj')
            alpha_range = (0, -3, 20)
            cache_file = join(cache_dir, args.qa_questions_version + '_' +
                              args.qa_embedding_model.replace('/', '-') + '_' + str(alpha_range) + '.joblib')
        else:
            # use hard-coded feature selection result from S03
            cache_dir = join(config.root_dir, 'qa', 'sparse_feats')
            alpha_range = (0, -3, 15)
            cache_file = join(
                cache_dir, 'v3_boostexamples_mistralai-Mistral-7B-Instruct-v0.2_(0, -3, 15).joblib')
            # 'v3_boostexamples_(0, -3, 15).joblib'
            # 'v3_boostexamples_mistralai-Mistral-7B-Instruct-v0.2_(0, -3, 15).joblib'
        os.makedirs(cache_dir, exist_ok=True)

        if os.path.exists(cache_file):
            alphas_enet, coefs_enet = joblib.load(cache_file)
            print('Loaded from cache:', cache_file)
        else:
            print('Couldn\'t find cache file:', cache_file, 'fitting now...')
            # get special resps by concatenating across subjects
            resp_train_shared = response_utils.get_resps_full(
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
    print('fitting regression...')
    r, model_params_to_save = fit_regression(
        args, r, stim_train_delayed, resp_train, stim_test_delayed, resp_test)

    # evaluate per voxel
    if args.pc_components > 0:
        resp_test = response_utils.load_response(
            story_names_test, args.subject)
        r['corrs_test'] = evaluate_pc_model_on_each_voxel(
            args, stim_test_delayed, resp_test,
            model_params_to_save, pca, scaler_test)
        # model_params_to_save['pca'] = pca
        model_params_to_save['scaler_test'] = scaler_test
        model_params_to_save['scaler_train'] = scaler_train

        # compute weighted corrs_tune_pc
        explained_var_weight = pca.explained_variance_[:args.pc_components]
        explained_var_weight = explained_var_weight / \
            explained_var_weight.sum() * len(explained_var_weight)
        r['corrs_tune_pc_weighted_mean'] = np.mean(
            explained_var_weight * r['corrs_tune_pc'])

    # add extra stats
    r = add_summary_stats(r, verbose=True)

    os.makedirs(save_dir_unique, exist_ok=True)
    joblib.dump(r, join(save_dir_unique, "results.pkl"))
    joblib.dump(model_params_to_save, join(
        save_dir_unique, "model_params.pkl"))
    print(
        f"Succesfully completed in {(time.time() - t0)/60:0.1f} minutes, saved to {save_dir_unique}")
