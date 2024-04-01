from config import mnt_dir
import itertools
import os
from os.path import dirname, join
import sys
from dict_hash import sha256
import subprocess
from imodelsx import submit_utils
path_to_file = os.path.dirname(os.path.abspath(__file__))
repo_dir = dirname(dirname(os.path.abspath(__file__)))
sys.path.append(repo_dir)
# python /home/chansingh/fmri/01_fit_encoding.py

params_shared_dict = {
    # 'pc_components': [1000, 100, -1],  # [5000, 100, -1], # default -1 predicts each voxel independently
    # 'pc_components': [100, 1000, -1],
    'pc_components': [100],
    'encoding_model': ['ridge'],


    # things to average over
    'use_cache': [1],
    # 'save_dir': ['/home/chansingh/mntv1/deep-fMRI/encoding/results_mar28'],
    'save_dir': ['/home/chansingh/mntv1/deep-fMRI/encoding/results_apr1'],
    'nboots': [5],

    # fixed params
    # 'UTS03', 'UTS01', 'UTS02'],
    'subject': ['UTS03'],
    'use_test_setup': [0],
    'qa_embedding_model': [
        # 'mistralai/Mistral-7B-v0.1',
        "mistralai/Mixtral-8x7B-v0.1"
    ],
}


# main args are qa_embedder-10, 'v2', seed=1, ndelays=8
params_coupled_dict = {
    ('feature_space', 'qa_questions_version', 'seed', 'ndelays'): [
        # baselines
        # ('bert-10', 'v1', 1, 4),
        # ('bert-10', 'v1', 2, 8),
        # ('bert-10', 'v1', 3, 12),
        # ('eng1000', 'v1', 1, 4),
        # ('eng1000', 'v1', 1, 8),
        # ('eng1000', 'v1', 1, 12),

        # # main
        # ('qa_embedder-10', 'v2', 1, 4),
        # ('qa_embedder-10', 'v2', 2, 8),
        # ('qa_embedder-10', 'v2', 3, 12),

        # # ablation ngrams (should have run this with v2 but never did)
        # ('qa_embedder-5', 'v1', 1, 4),
        # ('qa_embedder-5', 'v1', 2, 8),
        # ('qa_embedder-5', 'v1', 3, 12),

        # ablation version
        ('qa_embedder-10', 'v1', 1, 4),
        ('qa_embedder-10', 'v1', 2, 8),
        ('qa_embedder-10', 'v1', 3, 12),
        ('qa_embedder-10', 'v1', 4, 8),
        ('qa_embedder-10', 'v1', 5, 8),
        ('qa_embedder-10', 'v1', 6, 8),
        ('qa_embedder-10', 'v1', 7, 8),
        ('qa_embedder-10', 'v1', 8, 8),
        ('qa_embedder-10', 'v1', 9, 8),
        ('qa_embedder-10', 'v1', 10, 8),
        ('qa_embedder-10', 'v1', 11, 8),
        ('qa_embedder-10', 'v1', 12, 8),
        ('qa_embedder-10', 'v1', 13, 8),
        ('qa_embedder-10', 'v1', 14, 8),
        ('qa_embedder-10', 'v1', 15, 8),
        ('qa_embedder-10', 'v1', 16, 8),
        ('qa_embedder-10', 'v1', 17, 8),
        ('qa_embedder-10', 'v1', 18, 8),
        ('qa_embedder-10', 'v1', 19, 8),
        ('qa_embedder-10', 'v1', 20, 8),
        ('qa_embedder-10', 'v1', 21, 8),
        ('qa_embedder-10', 'v1', 22, 8),
        ('qa_embedder-10', 'v1', 23, 8),
        ('qa_embedder-10', 'v1', 24, 8),
        ('qa_embedder-10', 'v1', 25, 8),
    ],
}
# Args list is a list of dictionaries
# If you want to do something special to remove some of these runs, can remove them before calling run_args_list
args_list = submit_utils.get_args_list(
    params_shared_dict=params_shared_dict,
    params_coupled_dict=params_coupled_dict,
)
script_name = join(repo_dir, '01_fit_encoding.py')
amlt_kwargs = {
    'amlt_file': join(repo_dir, 'launch.yaml'),
    # [64G16-MI200-IB-xGMI, 64G16-MI200-xGMI
    # 'sku': '64G8-MI200-xGMI',
    # 'sku': '64G2-MI200-xGMI',
    'sku': '64G4-MI200-xGMI',
    'mnt_rename': ('/home/chansingh/mntv1', '/mntv1'),
}
submit_utils.run_args_list(
    args_list,
    script_name=script_name,
    actually_run=True,
    amlt_kwargs=amlt_kwargs,
    # gpu_ids=[0, 1],
    # n_cpus=9,
    # n_cpus=6,
    # gpu_ids=[0, 1, 2, 3],
    # gpu_ids=[0, 1, 2, 3],
    # gpu_ids=[[0, 1, 2, 3]],
    # gpu_ids=[[0, 1], [2, 3]],
    repeat_failed_jobs=True,
    shuffle=True,
)
