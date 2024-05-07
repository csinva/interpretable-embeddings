

import os
from os.path import dirname, join, expanduser
import sys
from imodelsx import submit_utils
path_to_file = os.path.dirname(os.path.abspath(__file__))
repo_dir = dirname(dirname(os.path.abspath(__file__)))
sys.path.append(repo_dir)
# python /home/chansingh/fmri/01_fit_encoding.py
MIST7B = 'mistralai/Mistral-7B-Instruct-v0.2'
MIXTMOE = 'mistralai/Mixtral-8x7B-Instruct-v0.1'
LLAMA8B = 'meta-llama/Meta-Llama-3-8B-Instruct'
LLAMA8B_fewshot = 'meta-llama/Meta-Llama-3-8B-Instruct-fewshot'
LLAMA70B_fewshot = 'meta-llama/Meta-Llama-3-70B-Instruct-fewshot2'
# (llama2-70B_lay24-10, 4 delays)

params_shared_dict = {
    # things to average over
    'use_cache': [1],
    'nboots': [5],
    'use_test_setup': [0],
    'encoding_model': ['ridge'],
    'subject': ['UTS03'],
    # 'subject': ['UTS02'],
    # 'subject': ['UTS01', 'UTS02', 'UTS03'],
    # 'distill_model_path': [BEST_RUN],
    'save_dir': ['/home/chansingh/mntv1/deep-fMRI/encoding/results_apr7'],
    # 'ndelays': [4, 8, 12],
    'ndelays': [8],
    'pc_components': [100],

    # feature selection...
    'num_stories': [0],  # this is used to get shared stories, only u
    'feature_selection_alpha_index': [1],
    # 'feature_selection_alpha_index': range(2, 10),
    # 'feature_selection_alpha_index': range(3, 11),

    # local
    # 'seed': [1],
    # 'pc_components': [1000, 100, -1],
    'use_extract_only': [0],
}

params_coupled_dict = {
    ('feature_space', 'qa_questions_version', 'qa_embedding_model'): [
        # new
        ('bert-10', 'v1', MIST7B),
        ('eng1000', 'v1', MIST7B),  # need to rerun sparsity for this...
        # run this with num_stories not 0 for old
        ('qa_embedder-10', 'v3_boostexamples', 'ensemble1'),
    ],
}
# Args list is a list of dictionaries
# If you want to do something special to remove some of these runs, can remove them before calling run_args_list
args_list = submit_utils.get_args_list(
    params_shared_dict=params_shared_dict,
    params_coupled_dict=params_coupled_dict,
)
script_name = join(repo_dir, 'experiments', '02_fit_encoding.py')
# amlt_kwargs = {
#     # 'amlt_file': join(repo_dir, 'scripts', 'launch_cpu.yaml'),
#     # 'sku': 'E4ads_v5',
#     # 'mnt_rename': ('/home/chansingh/mntv1', '/mntv1'),
#     'amlt_file': join(repo_dir, 'launch.yaml'),  # change this to run a cpu job
#     'sku': '64G2-MI200-xGMI',
#     'mnt_rename': ('/home/chansingh/mntv1', '/mntv1'),
# }
amlt_kwargs = {
    'amlt_file': join(repo_dir, 'scripts', 'launch_cpu.yaml'),
    # E4ads_v5 (30 GB), E8ads_v5 (56 GB), E16ads_v5 (120GB), E32ads_v5 (240GB), E64ads_v5 (480 GB)
    'sku': 'E64ads_v5',
    # 'sku': 'E32ads_v5',
    'mnt_rename': ('/home/chansingh/mntv1', '/mntv1'),
}
submit_utils.run_args_list(
    args_list,
    script_name=script_name,
    # unique_seeds='seed_stories',
    amlt_kwargs=amlt_kwargs,
    # n_cpus=9,
    # n_cpus=2,
    # gpu_ids=[0, 1],
    # gpu_ids=[0, 1, 2, 3],
    # gpu_ids=[[0, 1], [2, 3]],
    # gpu_ids=[[0, 1, 2, 3]],
    # actually_run=False,
    repeat_failed_jobs=True,
    shuffle=True,
    cmd_python=f'export HF_TOKEN={open(expanduser("~/.HF_TOKEN"), "r").read().strip()}; python',
)
