import cortex
from tqdm import tqdm
import joblib
import qa_questions
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
fit_encoding = __import__('01_fit_encoding')
path_to_repo = dirname(dirname(os.path.abspath(__file__)))
dvu.set_style()


def get_weights_top(args):
    model_params_to_save = joblib.load(
        join(args.save_dir_unique, 'model_params.pkl'))
    print(f'{args.feature_space=}, {args.pc_components=}, {args.ndelays=} {args.qa_embedding_model}')

    # get weights
    ndelays = args.ndelays
    weights = model_params_to_save['weights']
    assert weights.shape[0] % ndelays == 0
    emb_size = weights.shape[0] / ndelays
    weights = weights.reshape(ndelays, int(emb_size), -1)
    weights = weights.mean(axis=0)
    return weights


def save_coefs_csv(weights, out_dir, version):
    '''weights should be emb_size x num_voxels
    '''
    # look at coefs per feature
    weights = np.abs(weights)
    weights_per_feat = weights.mean(axis=-1)

    questions = qa_questions.get_questions(version, full=True)
    print(version, 'shapes', weights_per_feat.shape,
          weights.shape, len(questions))
    df = (
        pd.DataFrame({
            'question': questions,
            'avg_abs_coef_normalized': weights_per_feat / weights_per_feat.max()
        }).sort_values(by='avg_abs_coef_normalized', ascending=False)
        # .set_index('question')
        .round(3)
    )
    # df.to_json('../questions_v1.json', orient='index', indent=2)
    df.to_csv(join(out_dir, 'questions.csv'), index=False)
    return df


def save_coefs_flatmaps(weights, df, out_dir, subject='UTS03', num_flatmaps=10):
    '''weights should be emb_size x num_voxels
    '''
    for i in tqdm(range(num_flatmaps)):
        row = df.iloc[i]
        emb_dim_idx = row.name
        fname_save = join(out_dir, f'{i}___{row.question}.png')
        w = weights[emb_dim_idx]
        vabs = max(np.abs(w))
        vol = cortex.Volume(
            w, subject, xfmname=f'{subject}_auto', vmin=-vabs, vmax=vabs)
        cortex.quickshow(vol, with_rois=True, cmap='PuBu')
        plt.savefig(fname_save)
        plt.close()


if __name__ == '__main__':
    # select best model
    results_dir = '/home/chansingh/mntv1/deep-fMRI/encoding/results_apr1'

    # load the results in to a pandas dataframe
    r = imodelsx.process_results.get_results_df(results_dir)
    for k in ['save_dir', 'save_dir_unique']:
        r[k] = r[k].map(lambda x: x if x.startswith('/home')
                        else x.replace('/mntv1', '/home/chansingh/mntv1'))

    for version in ['v1', 'v4', 'v2', 'v3']:
        print('Version', version)
        out_dir = join(path_to_repo, 'qa_results', version)
        os.makedirs(out_dir, exist_ok=True)
        args = r[(r.feature_space == 'qa_embedder-10') *
                 #  (r.pc_components == -1) *
                 (r.pc_components == 100) *
                 (r.qa_questions_version == version)
                 ].sort_values(by='corrs_tune_mean', ascending=False).iloc[0]
        args_dict = {k: v for k, v in args.to_dict().items(
        ) if not isinstance(v, np.ndarray)}
        json.dump(args_dict, open(
            join(out_dir, 'meta.json'), 'w'), indent=2)

        weights = get_weights_top(args)
        df = save_coefs_csv(weights, out_dir, version=version)
        save_coefs_flatmaps(weights, df, out_dir,
                            subject='UTS03', num_flatmaps=10)
