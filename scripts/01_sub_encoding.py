import itertools
import os
from os.path import dirname, join
import sys
from imodelsx import submit_utils
path_to_file = os.path.dirname(os.path.abspath(__file__))
repo_dir = dirname(dirname(os.path.abspath(__file__)))
# python /home/chansingh/fmri/01_fit_encoding.py

params_shared_dict = {
    # things to vary
    # 'feature_space': [
    #     # 'gpt3-10', 'gpt3-20',
    #     'bert-10',
    #     'qa_embedder-5',
    #     # 'qa_embedder-10',
    #     # 'bert-20',
    #     'eng1000',
    #     # 'glove',
    #     # 'bert-3', 'bert-5',
    #     # 'roberta-10',
    #     # 'bert-sst2-10',
    # ],
    # default -1 predicts each voxel independently
    'pc_components': [1000, 100, -1],  # [5000, 100, -1],
    # 'pc_components': [100, -1],  # [5000, 100, -1],
    # 'encoding_model': ['mlp'],  # 'ridge'

    # things to average over
    'use_cache': [1],
    # 'save_dir': [join(repo_dir, 'results_mar27')],
    # 'save_dir': ['/home/chansingh/mntv1/deep-fMRI/encoding/results_mar27'],
    'save_dir': ['/home/chansingh/mntv1/deep-fMRI/encoding/results_mar28'],
    'nboots': [5, 10],
    # 'nboots': [50, 75],


    # fixed params
    # 'UTS03', 'UTS01', 'UTS02'],
    'subject': ['UTS03'],  # 'UTS01', 'UTS02'],
    # 'subject': ['UTS03'],
    # 'mlp_dim_hidden': [768],
    'use_test_setup': [0],
}
params_coupled_dict = {
    ('feature_space', 'seed', 'ndelays'): [
        ('bert-10', 1, 4),
        ('eng1000', 1, 4),
        ('qa_embedder-5', 1, 4),
        ('qa_embedder-5', 1, 8),
        ('qa_embedder-5', 1, 12),
    ],
}
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
    # gpu_ids=[0, 1],
    n_cpus=30,
    # gpu_ids=[0, 1, 2, 3],
    # gpu_ids=[[0, 1], [2, 3]],
    repeat_failed_jobs=True,
    shuffle=True,
)
