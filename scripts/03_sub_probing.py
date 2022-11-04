import itertools
import os
from os.path import dirname
import sys
import submit_utils
repo_dir = dirname(dirname(os.path.abspath(__file__)))
# python /home/chansingh/emb-gam/experimental/fmri/02_fit_decoding.py


PARAMS_COUPLED_DICT = {
    ('save_dir', 'subsample_frac'): [
        ('/home/chansingh/mntv1/deep-fMRI/results/linear_models/probing_nov2_sub_0.05', 0.05),
        # ('/home/chansingh/mntv1/deep-fMRI/results/linear_models/probing_oct28_sub_0.05', 0.05),
    ],
}

##########################################
# params shared across everything (higher up things are looped over first)
##########################################
PARAMS_SHARED_DICT = {
    # things to average over
    'seed': [1],  # , 2, 3],
    'perc_threshold_fmri': [0, 99],

    # things to vary
    'dset': [
        'probing-subj_number',
        'probing-obj_number',
        'probing-past_present', 'probing-sentence_length', 'probing-top_constituents',
        'probing-tree_depth', 'probing-coordination_inversion', 'probing-odd_man_out',
        'probing-bigram_shift',
        # 'probing-word_content', # this one has no positive examples
    ],
    'model': [
        'bert-10__ndel=4__pc=50000_fmri',
        'bert-10__ndel=4_fmri',
        'bert-base-uncased_embs',
    ],
}

ks_final, param_combos_final = submit_utils.combine_param_dicts(
    PARAMS_SHARED_DICT, PARAMS_COUPLED_DICT)

idx_model = ks_final.index('model')
idx_perc_threshold_fmri = ks_final.index('perc_threshold_fmri')

submit_utils.run_dicts(
    ks_final, param_combos_final,
    script_name='02_fit_decoding.py',
    actually_run=True,
    shuffle=True,
    reverse=False,
)
print('num combos', len(param_combos_final))
