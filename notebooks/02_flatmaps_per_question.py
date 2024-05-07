import cortex
from tqdm import tqdm
import joblib
import huth.features.qa_questions as qa_questions
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
sys.path.append('..')
fit_encoding = __import__('02_fit_encoding')
path_to_repo = dirname(dirname(os.path.abspath(__file__)))
dvu.set_style()


def get_weights_top(args):
    '''Return weights without delays out_size x
    '''

    model_params = joblib.load(
        join(args.save_dir_unique, 'model_params.pkl'))
    print(f'{args.feature_space=}, {args.pc_components=}, {args.ndelays=} {args.qa_embedding_model}')

    # get weights
    ndelays = args.ndelays
    weights = model_params['weights']
    assert weights.shape[0] % ndelays == 0
    emb_size = weights.shape[0] / ndelays
    weights = weights.reshape(ndelays, int(emb_size), -1)
    weights = weights.mean(axis=0)

    weights_pc = model_params['weights_pc']
    assert weights_pc.shape[0] % ndelays == 0
    qs_size = weights_pc.shape[0] / ndelays
    weights_pc = weights_pc.reshape(ndelays, int(qs_size), -1)
    weights_pc = weights_pc.mean(axis=0)

    return weights, weights_pc


def save_coefs_csv(weights, out_dir, version, questions):
    '''weights should be emb_size x num_voxels
    '''
    # look at coefs per feature
    weights = np.abs(weights)
    weights_per_feat = weights.mean(axis=-1)

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


def save_coefs_flatmaps(weights, df, out_dir, subject='S03', num_flatmaps=10):
    '''weights should be emb_size x num_voxels
    '''
    for i in tqdm(range(min(num_flatmaps, len(df)))):
        row = df.iloc[i]
        emb_dim_idx = row.name
        fname_save = join(out_dir, f'{i}___{row.question}.png')
        w = weights[emb_dim_idx]
        vabs = max(np.abs(w))
        vol = cortex.Volume(
            w, 'UT' + subject, xfmname=f'UT{subject}_auto', vmin=-vabs, vmax=vabs)
        cortex.quickshow(vol, with_rois=True, cmap='PuBu')
        plt.savefig(fname_save)
        plt.close()


if __name__ == '__main__':
    results_dir = '/home/chansingh/mntv1/deep-fMRI/encoding/results_apr7'
    for subject in ['S03', 'S02', 'S01']:
        # r = imodelsx.process_results.get_results_df(results_dir)
        r, cols_varied, mets = analyze_helper.load_clean_results(results_dir)
        r = r[r.distill_model_path.isna()]
        r = r[~(r.feature_space == 'qa_embedder-25')]
        r = r[r.pc_components == 100]
        r = r[~((r.feature_space == 'qa_embedder-10') &
                (r.qa_embedding_model != 'ensemble1'))]

        # for version in ['v1', 'v2', 'v3', 'v4']:
        for version in ['v3_boostexamples']:
            print('Version', version)
            args = r[(r.feature_space == 'qa_embedder-10') *
                     #  (r.pc_components == -1) *
                     (r.pc_components == 100) *
                     #  (r.qa_embedding_model == 'mist-7B') *
                     (r.qa_embedding_model == 'ensemble1') *
                     (r.qa_questions_version == version) *
                     (r.ndelays == 8) *
                     (r.subject == subject)
                     ]
            # args0 = args.sort_values(by='corrs_tune_pc_mean',
            # ascending=False).iloc[0]
            print(args[['feature_selection_alpha_index',
                        'weight_enet_mask_num_nonzero']])
            for feature_selection_alpha_index in sorted(args.feature_selection_alpha_index.unique(), reverse=False):
                # for feature_selection_alpha_index in [-1]:
                args0 = args[args.feature_selection_alpha_index ==
                             feature_selection_alpha_index].iloc[0]
                args_dict = {k: v for k, v in args0.to_dict().items(
                ) if not isinstance(v, np.ndarray)}

                weights, weights_pc = get_weights_top(args0)
                # emb_size x num_voxels
                questions = qa_questions.get_questions(version, full=True)
                if isinstance(args0.weight_enet_mask, np.ndarray):
                    questions = np.array(questions)[args0.weight_enet_mask]

                # save stuff
                out_dir = join(path_to_repo, 'qa_results', subject, version +
                               f'_num={len(questions)}')
                os.makedirs(out_dir, exist_ok=True)
                json.dump(args_dict, open(
                    join(out_dir, 'meta.json'), 'w'), indent=2)
                df = save_coefs_csv(
                    weights_pc, out_dir,
                    version=version,
                    questions=questions)

                # cap at 30 (weights is emb_size x num_voxels)
                num_flatmaps = min(len(questions), 40)
                joblib.dump(weights, join(out_dir, 'weights.pkl'))
                save_coefs_flatmaps(weights[:40], df, out_dir,
                                    subject=subject, num_flatmaps=num_flatmaps)
