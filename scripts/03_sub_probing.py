import itertools
import os
from os.path import dirname
import sys
import submit_utils
repo_dir = dirname(dirname(os.path.abspath(__file__)))
# python /home/chansingh/emb-gam/experimental/fmri/02_fit_decoding.py

PARAMS_COUPLED_DICT = {
    ('save_dir', 'subsample_frac'): [
        ('/home/chansingh/mntv1/deep-fMRI/results/linear_models/probing_oct25', -1),
    ],
}

##########################################
# params shared across everything (higher up things are looped over first)
##########################################
PARAMS_SHARED_DICT = {
    # things to average over
    'seed': [1],  # , 2, 3],
    'perc_threshold_fmri': [0],

    # things to vary
    'dset': [
        'probing-subj_number', 'probing-word_content', 'probing-obj_number',
        'probing-past_present', 'probing-sentence_length', 'probing-top_constituents',
        'probing-tree_depth', 'probing-coordination_inversion', 'probing-odd_man_out',
        'probing-bigram_shift',
    ],
    'model': [
        'bert-10__ndel=4fmri',
        'bert-base-uncased',
    ],
}

ks_final, param_combos_final = submit_utils.combine_param_dicts(
    PARAMS_SHARED_DICT, PARAMS_COUPLED_DICT)

idx_model = ks_final.index('model')
idx_perc_threshold_fmri = ks_final.index('perc_threshold_fmri')

# force idx_perc_threshold to 0 (default) unless it is an fmri model
param_combos_final = [
    p for p in param_combos_final
    if p[idx_model].endswith('fmri') or p[idx_perc_threshold_fmri] == 0
]

submit_utils.run_dicts(
    ks_final, param_combos_final,
    script_name='02_fit_decoding.py',
    actually_run=True,
    shuffle=False,
    reverse=False,
)
