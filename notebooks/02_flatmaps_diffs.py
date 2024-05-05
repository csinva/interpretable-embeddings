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
import analyze_helper
import sys
import json
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
sys.path.append('..')
path_to_repo = dirname(dirname(os.path.abspath(__file__)))


def _save_flatmap(vals, subject, fname_save, clab):
    vabs = max(np.abs(vals))
    cmap = 'RdBu'
    # cmap = sns.diverging_palette(12, 210, as_cmap=True)
    # cmap = sns.diverging_palette(16, 240, as_cmap=True)

    vol = cortex.Volume(
        vals, subject, xfmname=f'{subject}_auto', vmin=-vabs, vmax=vabs, cmap=cmap)

    cortex.quickshow(vol,
                     with_rois=False,
                     with_labels=False,
                     #  with_colorbar=True
                     )
    plt.savefig(fname_save)
    plt.close()

    # save cbar
    norm = Normalize(vmin=-vabs, vmax=vabs)
    # need to invert this to match above
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    fig, ax = plt.subplots(figsize=(5, 0.35))
    cbar = plt.colorbar(sm, cax=ax, orientation='horizontal')
    cbar.set_label(clab, fontsize='x-large')
    plt.savefig(fname_save.replace('flatmap.pdf',
                'cbar.pdf'), bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    results_dir = '/home/chansingh/mntv1/deep-fMRI/encoding/results_apr7'
    out_dir = join(path_to_repo, 'qa_results', 'diffs')
    os.makedirs(out_dir, exist_ok=True)

    # load the results in to a pandas dataframe
    r, cols_varied, mets = analyze_helper.load_clean_results(results_dir)
    r = r[r.feature_selection_alpha_index < 0]
    r = r[r.distill_model_path == 'None']
    r = r[~(r.feature_space == 'qa_embedder-25')]
    r = r[r.pc_components == 100]

    for subject in ['UTS03', 'UTS02', 'UTS01']:  # ['UTS01', 'UTS02', 'UTS03']:
        args_qa = r[
            (r.subject == subject) *
            (r.feature_space.str.contains('qa_embedder'))
        ].sort_values(by='corrs_tune_mean', ascending=False).iloc[0]
        for feature_space in ['qa_embedder', 'bert', 'llama']:
            corrs = []

            args_baseline = r[
                # (r.feature_space.str.contains('bert'))
                (r.feature_space.str.contains(feature_space)) *
                (r.subject == subject)
                # (r.ndelays == 8)
            ].sort_values(by='corrs_tune_mean', ascending=False).iloc[0]

            print('means', 'qa', args_qa['corrs_test'].mean(
            ), 'baseline', args_baseline['corrs_test'].mean())

            # fname_save = join(out_dir, f'diff_bert-qa.png')

            lab_name_dict = {
                'qa_embedder': 'QA-Emb',
                'bert': 'BERT',
                'llama': 'LLaMA'
            }
            clab = f'Test correlation ({lab_name_dict[feature_space]})'
            fname_save = join(
                out_dir, f'{subject}_{feature_space.replace("qa_embedder", "qa")}_flatmap.pdf')
            _save_flatmap(args_baseline['corrs_test'],
                          subject, fname_save, clab=clab)

            if not feature_space == 'qa_embedder':
                fname_save = join(
                    out_dir, f'{subject}_qa-{feature_space.replace("qa_embedder", "qa")}_flatmap.pdf')
                clab = f'Test correlation (QA-Emb - {lab_name_dict[feature_space]})'
                _save_flatmap(
                    args_qa['corrs_test'] - args_baseline['corrs_test'], subject, fname_save, clab=clab)
