import argparse
import datasets
import numpy as np
import os
from os.path import join
import logging
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
from transformers import pipeline
from ridge_utils.semantic_model import SemanticModel
from matplotlib import pyplot as plt
from typing import List, Tuple
from sklearn.linear_model import RidgeCV, LogisticRegressionCV, LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from feature_spaces import em_data_dir, data_dir, results_dir, nlp_utils_dir
import feature_spaces
from collections import defaultdict
import pandas as pd
import pickle as pkl
from sklearn import metrics
from copy import deepcopy
import sys
import data
from tqdm import tqdm
from sklearn import preprocessing


def get_word_vecs(X: List[str], model='eng1000') -> np.ndarray:
    if 'eng1000' in model:
        sm = SemanticModel.load(join(em_data_dir, 'english1000sm.hf5'))
    elif 'glove' in model:
        sm = SemanticModel.load_np(join(nlp_utils_dir, 'glove'))
    # extract features
    X = [
        [word.encode('utf-8') for word in sentence.split(' ')]
        for sentence in X
    ]
    feats = sm.project_stims(X)
    return feats


def get_bow_vecs(X: List[str], X_test: List[str]):
    trans = CountVectorizer().fit(X).transform
    return trans(X).todense(), trans(X_test).todense()


def get_embs_llm(X: List[str], checkpoint: str):  # X_test: List[str],
    """Return embeddings from HF model given checkpoint name
    (Fixed-size embedding by averaging over seq_len)
    """
    pipe = pipeline("feature-extraction",
                    model=checkpoint,
                    truncation=True,
                    device=0)

    def get_emb(x):
        return {'emb': pipe(x['text'])}
    text = datasets.Dataset.from_dict({'text': X})
    out_list = text.map(get_emb)['emb']
    # out_list is (batch_size, 1, (seq_len + 2), 768)

    # convert to np array by averaging over len (can't just convert the since seq lens vary)
    num_examples = len(out_list)
    dim_size = len(out_list[0][0][0])
    embs = np.zeros((num_examples, dim_size))
    logging.info('extract embs HF...')
    for i in tqdm(range(num_examples)):
        embs[i] = np.mean(out_list[i], axis=1)  # avg over seq_len dim
    return embs


def get_embs_fmri(X: List[str], model, save_dir_fmri, perc_threshold=98) -> np.ndarray:
    """Get initial embeddings then apply learned fMRI transform
    """
    if model.lower().startswith('bert-') or model.lower().startswith('roberta'):
        checkpoint = feature_spaces._FEATURE_CHECKPOINTS[model[:model.index(
            '__')]]
        feats = get_embs_llm(X, checkpoint)
        # ngram_size = int(model.split('-')[-1].split('__')[0])
        # feats = get_ngram_vecs(X, model=model)
    else:
        feats = get_word_vecs(X, model=model)
    feats = preprocessing.StandardScaler().fit_transform(feats)

    # load fMRI transform
    weights_npz = np.load(join(save_dir_fmri, 'weights.npz'))
    weights = weights_npz['arr_0']
    ndelays_str = model[model.index('ndel=') + len('ndel='):]
    if '__' in ndelays_str:
        ndelays_str = ndelays_str[:ndelays_str.index('__')]
    ndelays = int(ndelays_str)
    weights = weights.reshape(ndelays, -1, feats.shape[-1])
    weights = weights.mean(axis=0).squeeze()  # mean over delays dimension...

    # apply fMRI transform
    embs = feats @ weights.T

    # subselect repr
    if perc_threshold > 0:
        if 'pc=' in model:
            NUM_PCS = 50000
            embs = embs[:, :int(NUM_PCS * (100 - perc_threshold) / 100)]
        elif not 'pc=' in model:
            corrs_val = np.load(join(save_dir_fmri, 'corrs.npz'))['arr_0']
            perc = np.percentile(corrs_val, perc_threshold)
            idxs = (corrs_val > perc)
            # print('emb dim', idxs.sum(), 'val corr cutoff', perc)
            embs = embs[:, idxs]

    return embs


def get_feats(model: str, X: List[str], X_test: List[str],
              subject_fmri: str = 'UTS03', perc_threshold_fmri: int = 0,
              args=None) -> Tuple[np.ndarray, np.ndarray]:
    """Return both training and testing features
    """
    logging.info('Extracting features for ' + model)
    mod = model
    for k in ['_fmri', '_vecs', '_joint', '_embs']:
        mod = mod.replace(k, '')
    if model.endswith('_fmri') or model.endswith('_joint'):
        save_dir_fmri = join(results_dir, 'encoding', mod, subject_fmri)
        feats_train = get_embs_fmri(
            X, mod, save_dir_fmri, perc_threshold=perc_threshold_fmri)
        feats_test = get_embs_fmri(
            X_test, mod, save_dir_fmri, perc_threshold=perc_threshold_fmri)
    elif model.endswith('_vecs'):
        assert mod in ['bow', 'eng1000', 'glove']
        if mod == 'bow':
            feats_train, feats_test = get_bow_vecs(X, X_test)
        elif mod in ['eng1000', 'glove']:
            feats_train = get_word_vecs(X, model=mod)
            feats_test = get_word_vecs(X_test, model=mod)
    elif model.endswith('_embs'):  # HF checkpoint
        feats_train = get_embs_llm(X, checkpoint=mod)
        feats_test = get_embs_llm(X_test, checkpoint=mod)

    # also append llm embs to fmri embs from above
    if model.endswith('_joint'):
        checkpoint = feature_spaces._FEATURE_CHECKPOINTS[mod[:mod.index('__')]]
        feats_train = np.hstack(
            (feats_train, get_embs_llm(X, checkpoint=checkpoint)))
        feats_test = np.hstack(
            (feats_test, get_embs_llm(X_test, checkpoint=checkpoint)))
    return feats_train, feats_test


def fit_decoding(
        feats_train, y_train, feats_test, y_test,
        fname_save, args, frac_train_to_drop=0.1
):
    """Randomly fits to only 90% of training data
    """
    np.random.seed(args.seed)
    r = defaultdict(list)

    # fit model
    logging.info('Fitting logistic...')

    if args.subsample_frac is None or args.subsample_frac < 0:
        feats_train, feats_drop, y_train, y_drop = train_test_split(
            feats_train, y_train, test_size=frac_train_to_drop, random_state=args.seed)

    # m = LogisticRegressionCV(random_state=args.seed, cv=3)
    # m = LogisticRegressionCV(random_state=args.seed, refit=True, cv=cv) # with refit, should get better performance but no variance
    # m.fit(feats_train, y_train)

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=args.seed)
    param_grid = {'C': np.logspace(-4, 4, 10)}
    logistic = LogisticRegression(random_state=args.seed)
    m = GridSearchCV(logistic, param_grid, refit=True, cv=cv)
    m.fit(feats_train, y_train)

    # save stuff
    acc = m.score(feats_test, y_test)
    logging.info(f'acc {acc:0.2f}')
    args_dict = vars(args)
    for arg in args_dict:
        r[arg].append(args_dict[arg])

    r['acc_cv'].append(m.best_score_)
    r['acc'].append(acc)
    # r['roc_auc'].append(metrics.roc_auc_score(y_test, m.predict(feats_test)))
    r['feats_dim'].append(feats_train.shape[1])

    df = pd.DataFrame.from_dict(r).set_index('model')
    df.to_pickle(fname_save)
    fname_head_tail = os.path.split(fname_save)
    pkl.dump(
        m, open(join(fname_head_tail[0], 'coef_' + fname_head_tail[1]), 'wb'))
    return df


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=str, default='UTS03')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--perc_threshold_fmri', type=int, default=0,
                        help='[0, 100] - percentile threshold for fMRI features \
                            if using PCs, this is the percentage of PCs to use. \
                            higher uses less features')
    parser.add_argument("--save_dir", type=str, default='/home/chansingh/.tmp')
    parser.add_argument('--model', type=str, default='glovevecs',
                        help='Which model to extract features with. \
                                *_vecs uses a word-vector model. \
                                *_embs uses LLM embeddings. \
                                *_fmri uses model finetuned on fMRI. \
                                *_joint uses fmri + hf model.')  # glovevecs, bert-10__ndel=4fmri, bert-base-uncased
    parser.add_argument('--dset', type=str, default='rotten_tomatoes',
                        choices=[
                            'trec', 'emotion', 'rotten_tomatoes', 'tweet_eval',
                            'sst2', 'go_emotions', 'poem_sentiment', 'moral_stories',
                            'ethics-commonsense', 'ethics-deontology', 'ethics-justice', 'ethics-utilitarianism', 'ethics-virtue',
                            'probing-subj_number', 'probing-obj_number',
                            'probing-past_present', 'probing-sentence_length', 'probing-top_constituents',
                            'probing-tree_depth', 'probing-coordination_inversion', 'probing-odd_man_out',
                            'probing-bigram_shift',  # 'probing-word_content',
                        ])
    # parser.add_argument('--use_normalized_embs_fmri', type=int, default=1,
    # help='whether to normalize embeddings before projecting to fmri space')
    parser.add_argument('--use_normalized_feats', type=int, default=0,
                        help='whether to normalize features before fitting')
    parser.add_argument('--nonlinearity', type=str, default=None,
                        help='pointwise nonlinearity for features')
    parser.add_argument('--subsample_frac', type=float,
                        default=None, help='fraction of data to use for training. If none or negative, use all the data')
    parser.add_argument('--pc_components', type=int, default=-1)
    parser.add_argument('--use_cache', type=int,
                        default=True, help='whether to use cache')
    return parser


def apply_pointwise_nonlinearity(feats_train, feats_test, nonlinearity='relu'):
    if nonlinearity is None:
        return feats_train, feats_test
    elif nonlinearity == 'sigmoid':
        def f(x): return 1 / (1 + np.exp(-x))
        return f(feats_train), f(feats_test)
    elif nonlinearity == 'relu':
        return np.clip(feats_train, a_min=0, a_max=None), np.clip(feats_test, a_min=0, a_max=None)
    elif nonlinearity == 'tanh':
        return np.tanh(feats_train), np.tanh(feats_test)



if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()

    # set up logging
    logger = logging.getLogger()
    logging.basicConfig(level=logging.INFO)
    # logging.basicConfig(level=logging.DEBUG)
    for k in sorted(vars(args)):
        logger.info('\t' + k + ' ' + str(vars(args)[k]))
    mod = args.model
    assert mod.endswith('_embs') or mod.endswith(
        '_fmri') or mod.endswith('_vecs') or mod.endswith('_joint')

    # check for caching
    def get_fname_save(args):
        fname_save = join(
            args.save_dir,
            f'{args.dset.replace("/", "-")}_{args.model}_perc={args.perc_threshold_fmri}_seed={args.seed}')
        if args.nonlinearity is not None:
            fname_save += f'_nonlin={args.nonlinearity}'
        return fname_save + '.pkl'
    fname_save = get_fname_save(args)
    if os.path.exists(fname_save) and args.use_cache:
        logging.info('\nAlready ran ' + fname_save + '!')
        logging.info('Skipping :)!\n')
        sys.exit(0)

    # get data
    X_train, y_train, X_test, y_test = data.get_dsets(
        args.dset, seed=args.seed, subsample_frac=args.subsample_frac)
    logging.debug(
        f'\data shape: {len(X_train)} {len(X_train[0])} {len(y_train)}')

    # fit decoding
    os.makedirs(args.save_dir, exist_ok=True)
    feats_train, feats_test = get_feats(
        args.model, X_train, X_test,
        subject_fmri=args.subject, perc_threshold_fmri=args.perc_threshold_fmri, args=args)
    if args.nonlinearity in ['sigmoid', 'relu', 'tanh']:
        feats_train, feats_test = apply_pointwise_nonlinearity(
            feats_train, feats_test, nonlinearity=args.nonlinearity)
    if args.use_normalized_feats:
        scaler = preprocessing.StandardScaler().fit(feats_train)
        feats_train = scaler.transform(feats_train)
        feats_test = scaler.transform(feats_test)
    fit_decoding(feats_train, y_train,
                 feats_test, y_test, fname_save, args)
    logging.info('Succesfully completed! saved to ' + fname_save)
