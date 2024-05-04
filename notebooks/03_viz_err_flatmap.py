import cortex
from tqdm import tqdm
import joblib
import imodelsx.process_results
import numpy as np
from os.path import join
from matplotlib import pyplot as plt
from copy import deepcopy
import pandas as pd
import os
from os.path import dirname
import seaborn as sns
import dvu
import sys
import json
sys.path.append('..')
path_to_repo = dirname(dirname(os.path.abspath(__file__)))


if __name__ == '__main__':
    # select best model
    results_dir = '/home/chansingh/mntv1/deep-fMRI/encoding/results_apr7'

    # load the results in to a pandas dataframe
    r = imodelsx.process_results.get_results_df(results_dir)
    for k in ['save_dir', 'save_dir_unique']:
        r[k] = r[k].map(lambda x: x if x.startswith('/home')
                        else x.replace('/mntv1', '/home/chansingh/mntv1'))

    corrs = []
    args0 = r[
        (r.feature_space.str.contains('qa_embedder')) *
        #  (r.pc_components == -1) *
        (r.pc_components == 100) *
        (r.ndelays == 8)
    ].sort_values(by='corrs_tune_mean', ascending=False).iloc[0]
    corrs.append(args0['corrs_test'])
    args_baseline = r[
        # (r.feature_space.str.contains('bert'))
        (r.feature_space.str.contains('llama2'))
        # (r.pc_components == 100)
        # (r.ndelays == 8)
    ].sort_values(by='corrs_tune_mean', ascending=False).iloc[0]
    corrs.append(args_baseline['corrs_test'])
    print('means', 'qa', corrs[0].mean(), 'baseline', corrs[1].mean())

    # diff = corrs[1] - corrs[0]
    # diff = corrs[0]
    diff = corrs[1]

    # save flatmap
    subject = args0.subject
    out_dir = join(path_to_repo, 'qa_results', 'corr_flatmaps')
    os.makedirs(out_dir, exist_ok=True)
    # fname_save = join(out_dir, f'diff_bert-qa.png')
    fname_save = join(out_dir, f'qa.png')
    vabs = max(np.abs(diff))
    vol = cortex.Volume(
        diff, subject, xfmname=f'{subject}_auto', vmin=-vabs, vmax=vabs)
    cortex.quickshow(vol, with_rois=True, cmap='PuBu')
    plt.savefig(fname_save)
    plt.close()
