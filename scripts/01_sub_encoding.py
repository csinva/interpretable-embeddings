import os
from os.path import dirname, join, expanduser
import sys
from imodelsx import submit_utils
path_to_file = os.path.dirname(os.path.abspath(__file__))
repo_dir = dirname(dirname(os.path.abspath(__file__)))
sys.path.append(repo_dir)
# python /home/chansingh/fmri/01_fit_encoding.py

params_shared_dict = {
    # things to average over
    'use_cache': [1],
    'save_dir': ['/home/chansingh/mntv1/deep-fMRI/encoding/results_apr7'],

    # fixed params
    'subject': ['UTS03'],
    'nboots': [5],
    'use_test_setup': [0],
    'encoding_model': ['ridge'],
    'ndelays': [4, 8, 12],

    # cluster
    # 'seed': range(14),
    # 'seed': range(20),
    'pc_components': [100],

    # local
    'seed': [1],
    # 'pc_components': [1000, 100, -1],
    # 'pc_components': [100],
    # 'use_extract_only': [0],
}

params_coupled_dict = {
    ('feature_space', 'qa_questions_version', 'qa_embedding_model'): [
        # # baselines
        # ('bert-10', 'v1', 'mistralai/Mistral-7B-Instruct-v0.2'),
        # ('eng1000', 'v1', 'mistralai/Mistral-7B-Instruct-v0.2'),

        # # # main
        # ('qa_embedder-10', 'v1', 'mistralai/Mistral-7B-Instruct-v0.2'),

        # # # vary mistral versions
        # ('qa_embedder-10', 'v2', 'mistralai/Mistral-7B-Instruct-v0.2'),
        # ('qa_embedder-10', 'v3', 'mistralai/Mistral-7B-Instruct-v0.2'),
        # ('qa_embedder-10', 'v4', 'mistralai/Mistral-7B-Instruct-v0.2'),
        ('qa_embedder-10', 'v5', 'mistralai/Mistral-7B-Instruct-v0.2'),

        # vary context len
        # ('qa_embedder-25', 'v1', 'mistralai/Mistral-7B-Instruct-v0.2'),

        # # mixtral
        # ('qa_embedder-10', 'v1', 'mistralai/Mixtral-8x7B-Instruct-v0.1'),
        # ('qa_embedder-10', 'v2', 'mistralai/Mixtral-8x7B-Instruct-v0.1'),
        # ('qa_embedder-10', 'v3', 'mistralai/Mixtral-8x7B-Instruct-v0.1'),
        # ('qa_embedder-10', 'v4', 'mistralai/Mixtral-8x7B-Instruct-v0.1'),

        # -last, -end versions (try 10, 50, 75)
        # ('qa_embedder-25', 'v1-last', 'mistralai/Mistral-7B-Instruct-v0.2'),
        # ('qa_embedder-25', 'v1-ending', 'mistralai/Mistral-7B-Instruct-v0.2'),
        # mixtral -last, -end.....
        # ('qa_embedder-25', 'v1-ending', 'mistralai/Mixtral-8x7B-Instruct-v0.1'),

        # bert sec versions
        # ('bert-sec3', 'v1', 'mistralai/Mistral-7B-Instruct-v0.2'),
        # ('bert-sec5', 'v1', 'mistralai/Mistral-7B-Instruct-v0.2'),

        # tr versions
        # ('bert-tr2', 'v1', 'mistralai/Mistral-7B-Instruct-v0.2'),
        # ('bert-tr3', 'v1', 'mistralai/Mistral-7B-Instruct-v0.2'),

        # qa sec versions
        # ('qa_embedder-sec3', 'v1', 'mistralai/Mistral-7B-Instruct-v0.2'),
        # ('qa_embedder-sec5', 'v1', 'mistralai/Mistral-7B-Instruct-v0.2'),

        # qa tr versions
        # ('qa_embedder-tr2', 'v1', 'mistralai/Mistral-7B-Instruct-v0.2'),
        # ('qa_embedder-tr3', 'v1', 'mistralai/Mistral-7B-Instruct-v0.2'),
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
    # 'sku': '64G4-MI200-xGMI',
    'sku': '64G2-MI200-xGMI',
    'mnt_rename': ('/home/chansingh/mntv1', '/mntv1'),
}
submit_utils.run_args_list(
    args_list,
    script_name=script_name,
    actually_run=True,
    # unique_seeds=True,
    # amlt_kwargs=amlt_kwargs,
    # n_cpus=9,
    # n_cpus=4,
    # gpu_ids=[0, 1],
    # gpu_ids=[0, 1, 2, 3],
    # gpu_ids=[[0, 1, 2, 3]],
    # gpu_ids=[[0, 1], [2, 3]],
    repeat_failed_jobs=True,
    shuffle=True,
    cmd_python=f'export HF_TOKEN={open(expanduser("~/.HF_TOKEN"), "r").read().strip()}; python',
)
