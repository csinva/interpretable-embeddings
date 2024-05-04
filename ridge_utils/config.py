from os.path import join, expanduser
# join(dirname(dirname(os.path.abspath(__file__))))

if 'chansingh' in expanduser('~'):
    mnt_dir = '/home/chansingh/mntv1'
else:
    mnt_dir = '/mntv1'

repo_dir = join(mnt_dir, 'deep-fMRI')
cache_embs_dir = join(repo_dir, 'cache_embs')
nlp_utils_dir = '/home/chansingh/nlp_utils'
em_data_dir = join(repo_dir, 'em_data')
data_dir = repo_dir  # join(repo_dir, 'data')
results_dir = join(repo_dir, 'results_new')
