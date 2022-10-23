import itertools
import os
from os.path import dirname
import sys
import submit_utils
repo_dir = dirname(dirname(os.path.abspath(__file__)))
# python /home/chansingh/emb-gam/experimental/fmri/02_fit_decoding.py

PARAMS_COUPLED_DICT = {
    ('save_dir', 'subsample_frac'): [
        ('/home/chansingh/mntv1/deep-fMRI/results/linear_models/oct23', -1),
        # ('/home/chansingh/mntv1/deep-fMRI/results/linear_models/subsamp_oct22', 0.1),
    ],
}

##########################################
# params shared across everything (higher up things are looped over first)
##########################################
PARAMS_SHARED_DICT = {
    # things to average over
    'seed': [1, 2, 3],
    'perc_threshold_fmri': [0.98, 0.99, 0.95],

    # things to vary
    'dset': [
        'rotten_tomatoes', 'moral_stories', 'sst2',
        'tweet_eval', 'emotion',
        'go_emotions', 'trec', 
    ],
    'model': [
        # 'roberta-10__ndel=4fmri',
        'bert-10__ndel=4fmri',
        'glove__ndel=4fmri',

        # 'roberta-large', 
        'bert-base-uncased',
        'glovevecs',
        # 'eng1000__ndel=4fmri',
        # 'eng1000vecs', 'bowvecs',
    ],
}

ks_final, param_combos_final = submit_utils.combine_param_dicts(
    PARAMS_SHARED_DICT, PARAMS_COUPLED_DICT)

idx_model = ks_final.index('model')
idx_perc_threshold_fmri = ks_final.index('perc_threshold_fmri')

# force idx_perc_threshold to 0.98 (default) unless it is an fmri model
param_combos_final = [
    p for p in param_combos_final
    if p[idx_model].endswith('fmri') or p[idx_perc_threshold_fmri] == 0.98
]

submit_utils.run_dicts(
    ks_final, param_combos_final,
    script_name='02_fit_decoding.py',
    actually_run=True,
    shuffle=True,
    reverse=False,
)
