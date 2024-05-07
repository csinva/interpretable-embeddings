from ridge_utils.data.response_utils import load_pca
import imodelsx.process_results
import viz
import dvu
from tqdm import tqdm
import seaborn as sns
import os
import pandas as pd
from collections import defaultdict
from matplotlib import pyplot as plt
from os.path import join
import numpy as np
import sys
sys.path.append('../experiments')
dvu.set_style()
fit_encoding = __import__('02_fit_encoding')
best_results_dir = '/home/chansingh/mntv1/deep-fMRI/qa/results/results_apr7'


def load_results(save_dir):
    dfs = []
    fnames = [
        fname for fname in os.listdir(save_dir)[::-1]
        if not fname.startswith('coef')
    ]
    for fname in tqdm(fnames):
        df = pd.read_pickle(join(save_dir, fname))
        # print(fname)
        # display(df)
        dfs.append(df.reset_index())
    d = pd.concat(dfs)
    # d = d.drop(columns='coef_')
    # .round(2)
    # d.set_index(['feats', 'dset'], inplace=True)
    d['nonlin_suffix'] = d['nonlinearity'].fillna(
        '').str.replace('None', '').str.replace('tanh', '_tanh')
    d['model'] = d['model'] + d['nonlin_suffix']
    d['model_full'] = d['model'] + '_thresh=' + \
        d['perc_threshold_fmri'].astype(str)
    return d


def load_clean_results(results_dir, experiment_filename='../experiments/02_fit_encoding.py'):
    # load the results in to a pandas dataframe
    r = imodelsx.process_results.get_results_df(results_dir)
    r = imodelsx.process_results.fill_missing_args_with_default(
        r, experiment_filename)
    for k in ['save_dir', 'save_dir_unique']:
        r[k] = r[k].map(lambda x: x if x.startswith('/home')
                        else x.replace('/mntv1', '/home/chansingh/mntv1'))
    r['qa_embedding_model'] = r.apply(lambda row: {
        'mistralai/Mistral-7B-Instruct-v0.2': 'mist-7B',
        'mistralai/Mixtral-8x7B-Instruct-v0.1': 'mixt-moe',
        'meta-llama/Meta-Llama-3-8B-Instruct': 'llama3-8B',
        'meta-llama/Meta-Llama-3-8B-Instruct-fewshot': 'llama3-8B-fewshot',
        'meta-llama/Meta-Llama-3-8B-Instruct-refined': 'llama3-8B-refined',
    }.get(row['qa_embedding_model'], row['qa_embedding_model']) if 'qa_emb' in row['feature_space'] else '', axis=1)
    r['subject'] = r['subject'].str.replace('UTS', 'S')
    r['qa_questions_version'] = r.apply(
        lambda row: row['qa_questions_version'] if 'qa_emb' in row['feature_space'] else 'eng1000', axis=1)
    mets = [c for c in r.columns if 'corrs' in c and (
        'mean' in c or 'frac' in c)]
    cols_varied = imodelsx.process_results.get_experiment_keys(
        r, experiment_filename)
    print('experiment varied these params:', cols_varied)
    r['corrs_test_mean_sem'] = r['corrs_test'].apply(
        lambda x: np.std(x) / np.sqrt(len(x)))
    mets.append('corrs_test_mean_sem')
    return r, cols_varied, mets


def add_corrs_tune_pc_weighted(r):
    r['corrs_tune_pc_weighted_mean'] = np.nan
    for subject in ['S01', 'S02', 'S03']:
        pca = load_pca('UT' + subject, pc_components=100)
        explained_var_weight = pca.explained_variance_[:100]
        explained_var_weight = explained_var_weight / \
            explained_var_weight.sum() * len(explained_var_weight)

        for i, row in r[r.subject == subject].iterrows():
            corrs = row['corrs_tune_pc']
            corrs_weighted = corrs * explained_var_weight
            r.loc[i, 'corrs_tune_pc_weighted_mean'] = corrs_weighted.sum()
    return r
