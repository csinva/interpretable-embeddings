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
LLAMA70B_fewshot = 'meta-llama/Meta-Llama-3-70B-Instruct-fewshot'
LLAMA70B = 'meta-llama/Meta-Llama-3-70B-Instruct'
BEST_RUN = '/home/chansingh/mntv1/deep-fMRI/encoding/results_apr7/68936a10a548e2b4ce895d14047ac49e7a56c3217e50365134f78f990036c5f7'

params_shared_dict = {
    # things to average over
    'use_cache': [1],
    'nboots': [5],
    'use_test_setup': [0],
    'encoding_model': ['ridge'],
    # 'subject': ['UTS03'],
    'subject': ['UTS03', 'UTS02', 'UTS01'],
    # 'subject': ['UTS01', 'UTS02'],
    'save_dir': ['/home/chansingh/mntv1/deep-fMRI/encoding/results_apr7'],
    'ndelays': [4, 8, 12],
    # 'ndelays': [4],
    # 'ndelays': [12],

    # cluster
    # 'seed_stories': range(6),
    'pc_components': [100],
    # 'ndelays': [4],

    # local
    # 'seed': range(5),
    # 'pc_components': [1000, 100, -1],
    # 'use_extract_only': [0],
}

params_coupled_dict = {
    ('feature_space', 'qa_questions_version', 'qa_embedding_model'): [
        # # baselines
        # ('bert-10', 'v1', MIST7B),
        # ('eng1000', 'v1', MIST7B),
        # ('finetune_roberta-base-10', 'v1', MIST7B),
        ('finetune_roberta-base_binary-10', 'v1', MIST7B),

        # ('llama2-7B_lay6-10', 'v1', MIST7B),
        # ('llama2-7B_lay12-10', 'v1', MIST7B),
        # ('llama2-7B_lay18-10', 'v1', MIST7B),
        # ('llama2-7B_lay24-10', 'v1', MIST7B),
        # ('llama2-7B_lay30-10', 'v1', MIST7B),

        # ('llama2-70B_lay12-10', 'v1', MIST7B),
        # ('llama2-70B_lay24-10', 'v1', MIST7B),  # this is best one
        # ('llama2-70B_lay36-10', 'v1', MIST7B),
        # ('llama2-70B_lay48-10', 'v1', MIST7B),
        # ('llama2-70B_lay60-10', 'v1', MIST7B),

        # ('llama3-8B_lay6-10', 'v1', MIST7B),
        # ('llama3-8B_lay12-10', 'v1', MIST7B),
        # ('llama3-8B_lay18-10', 'v1', MIST7B),
        # ('llama3-8B_lay24-10', 'v1', MIST7B),
        # ('llama3-8B_lay30-10', 'v1', MIST7B),

        # # # main
        # ('qa_embedder-10', 'v1', LLAMA8B),


        # ensemble
        # ('qa_embedder-10', 'v1', 'ensemble1'),
        # ('qa_embedder-10', 'v2', 'ensemble1'),
        # ('qa_embedder-10', 'v3_boostexamples', 'ensemble1'),
        # ('qa_embedder-10', 'v3', 'ensemble1'),




        # vary question versions
        # ('qa_embedder-10', 'v1', MIST7B),
        # ('qa_embedder-10', 'v2', MIST7B),
        # ('qa_embedder-10', 'v3', MIST7B),
        # ('qa_embedder-10', 'v4', MIST7B),
        # ('qa_embedder-10', 'v5', MIST7B),
        # ('qa_embedder-10', 'v6', MIST7B),
        # ('qa_embedder-10', 'v3_boostbasic', MIST7B),
        # ('qa_embedder-10', 'v3_boostexamples', MIST7B),
        # ('qa_embedder-10', 'v4_boostexamples', MIST7B),

        # # llama/mixtral
        # ('qa_embedder-10', 'v2', LLAMA8B),
        # ('qa_embedder-10', 'v3_boostexamples', LLAMA8B),
        # ('qa_embedder-10', 'v4_boostexamples', LLAMA8B),
        # ('qa_embedder-10', 'v1', LLAMA8B_fewshot),
        # ('qa_embedder-10', 'v2', LLAMA8B_fewshot),
        # ('qa_embedder-10', 'v3_boostexamples', LLAMA8B_fewshot),
        # ('qa_embedder-10', 'v1', LLAMA70B),
        # ('qa_embedder-10', 'v1', LLAMA70B_fewshot),
        # ('qa_embedder-10', 'v1', 'meta-llama/Meta-Llama-3-8B-Instruct-refined'),
        # ('qa_embedder-10', 'v2', 'meta-llama/Meta-Llama-3-8B-Instruct-refined'),
        # ('qa_embedder-10', 'v3_boostexamples',
        #  'meta-llama/Meta-Llama-3-8B-Instruct-refined'),
        # ('qa_embedder-10', 'v3',
        #  'meta-llama/Meta-Llama-3-8B-Instruct-refined'),

    ],
}
# Args list is a list of dictionaries
# If you want to do something special to remove some of these runs, can remove them before calling run_args_list
args_list = submit_utils.get_args_list(
    params_shared_dict=params_shared_dict,
    params_coupled_dict=params_coupled_dict,
)
script_name = join(repo_dir, '02_fit_encoding.py')
amlt_kwargs = {
    'amlt_file': join(repo_dir, 'launch.yaml'),  # change this to run a cpu job
    # [64G16-MI200-IB-xGMI, 64G16-MI200-xGMI
    # 'sku': '64G8-MI200-xGMI',
    # 'sku': '64G4-MI200-xGMI',
    'sku': '64G2-MI200-xGMI',
    'mnt_rename': ('/home/chansingh/mntv1', '/mntv1'),
}
amlt_kwargs_cpu = {
    'amlt_file': join(repo_dir, 'launch_cpu.yaml'),
    # E4ads_v5 (30 GB), E8ads_v5 (56 GB), E16ads_v5 (120GB), E32ads_v5 (240GB), E64ads_v5 (480 GB)
    # 'sku': 'E64ads_v5',
    'sku': 'E32ads_v5',
    'mnt_rename': ('/home/chansingh/mntv1', '/mntv1'),
}
submit_utils.run_args_list(
    args_list,
    script_name=script_name,
    unique_seeds='seed_stories',
    # amlt_kwargs=amlt_kwargs,
    # amlt_kwargs=amlt_kwargs_cpu,
    # n_cpus=9,
    n_cpus=3,
    # gpu_ids=[0, 1],
    # gpu_ids=[0, 1, 2, 3],
    # gpu_ids=[[0, 1], [2, 3]],
    # gpu_ids=[[0, 1, 2, 3]],
    # actually_run=False,
    repeat_failed_jobs=True,
    shuffle=True,
    cmd_python=f'export HF_TOKEN={open(expanduser("~/.HF_TOKEN"), "r").read().strip()}; python',
)


##################################### abandoned sweeps ##########################
# mixtral
# ('qa_embedder-10', 'v1', MIXTMOE),
# ('qa_embedder-10', 'v2', MIXTMOE),
# ('qa_embedder-10', 'v3', MIXTMOE),
# ('qa_embedder-10', 'v4', MIXTMOE),

# vary context len
# ('qa_embedder-25', 'v1', MIST7B),

# -last, -end versions (try 10, 50, 75)
# ('qa_embedder-25', 'v1-last', MIST7B),
# ('qa_embedder-25', 'v1-ending', MIST7B),
# mixtral -last, -end.....
# ('qa_embedder-25', 'v1-ending', MIXTMOE),

# bert sec versions
# ('bert-sec3', 'v1', MIST7B),
# ('bert-sec5', 'v1', MIST7B),

# tr versions
# ('bert-tr2', 'v1', MIST7B),
# ('bert-tr3', 'v1', MIST7B),

# qa sec versions
# ('qa_embedder-sec3', 'v1', MIST7B),
# ('qa_embedder-sec5', 'v1', MIST7B),

# qa tr versions
# ('qa_embedder-tr2', 'v1', MIST7B),
# ('qa_embedder-tr3', 'v1', MIST7B),
