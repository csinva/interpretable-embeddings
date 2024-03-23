import itertools
import os
from os.path import dirname, join
import sys
from imodelsx import submit_utils
path_to_file = os.path.dirname(os.path.abspath(__file__))
repo_dir = dirname(dirname(os.path.abspath(__file__)))
# python /home/chansingh/emb-gam/experimental/fmri/01_fit_encoding.py --feature bert-10 --ndelays 2 --seed 1 --subject UTS03

##########################################
# params shared across everything (higher up things are looped over first)
##########################################
params_shared_dict = {
    # things to vary
    'ndelays': [4],
    'feature_space': [
        # 'gpt3-10', 'gpt3-20',
        # 'bert-10',
        'qa_embedder-5',
        'qa_embedder-10',
        # 'bert-20',
        # 'eng1000', 'glove',
        # 'bert-3', 'bert-5',
        # 'roberta-10',
        # 'bert-sst2-10',
    ],
    # -1, 50000
    # 'pc_components': [10],  # default -1 predicts each voxel independently
    # 'encoding_model': ['mlp'],  # 'ridge'

    # things to average over
    'seed': [1, 2, 3, 4],
    'use_cache': [1],
    'save_dir': [join(repo_dir, 'results')],

    # fixed params
    # 'UTS03', 'UTS01', 'UTS02'],
    'subject': ['UTS03'],  # , 'UTS04', 'UTS05', 'UTS06'],
    # 'mlp_dim_hidden': [768],
}
params_coupled_dict = {}
# Args list is a list of dictionaries
# If you want to do something special to remove some of these runs, can remove them before calling run_args_list
args_list = submit_utils.get_args_list(
    params_shared_dict=params_shared_dict,
    params_coupled_dict=params_coupled_dict,
)
submit_utils.run_args_list(
    args_list,
    script_name=join(repo_dir, '01_fit_encoding.py'),
    actually_run=True,
    gpu_ids=[0, 1, 2, 3],
    shuffle=True,
)
