import itertools
import os
from os.path import dirname
import sys
import submit_utils
repo_dir = dirname(dirname(os.path.abspath(__file__)))
# python /home/chansingh/emb-gam/experimental/fmri/01_fit_encoding.py --feature bert-10 --ndelays 2 --seed 1 --subject UTS03

PARAMS_COUPLED_DICT = {}

##########################################
# params shared across everything (higher up things are looped over first)
##########################################
PARAMS_SHARED_DICT = {
    # things to vary
    'ndelays': [4],
    'feature': [
        # 'gpt3-10', 'gpt3-20',
        'bert-10', 'bert-20',
        # 'eng1000', 'glove',
        # 'bert-3', 'bert-5',
        # 'roberta-10', 
        # 'bert-sst2-10',
    ],
    # -1, 50000
    'pc_components' : [10], # default -1 predicts each voxel independently
    'encoding_model': ['mlp'], # 'ridge'

    # things to average over
    'seed': [1],
    'use_cache': [0],

    # fixed params
    # 'UTS03', 'UTS01', 'UTS02'],
    'subject': ['UTS03'], #, 'UTS04', 'UTS05', 'UTS06'],
    'mlp_dim_hidden': [768],
}

ks_final, param_combos_final = submit_utils.combine_param_dicts(
    PARAMS_SHARED_DICT, PARAMS_COUPLED_DICT)
submit_utils.run_dicts(
    ks_final, param_combos_final,
    script_name='01_fit_encoding.py',
    actually_run=True,
)
