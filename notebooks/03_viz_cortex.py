import cortex
import numpy as np
from os.path import join
from cortex.options import config
import h5py
import pickle as pkl
from copy import deepcopy
import matplotlib.pyplot as plt


encoding_result = '/home/chansingh/mntv1/deep-fMRI/results/encoding/bert-10__ndel=4/UTS03/'
decoding_result = '/home/chansingh/mntv1/deep-fMRI/results/linear_models/oct25'
decoding_fname = 'coef_rotten_tomatoes_bert-10__ndel=4fmri_perc=0_seed=1.pkl'
coef_fname = join(decoding_result, decoding_fname)

# load test corrs for all voxels
corrs = np.load(join(encoding_result, 'corrs.npz'))['arr_0']
corrs.shape

# find test corrs above a certain percentile
# this is what was used to train the decoding model
perc_threshold = 0
perc = np.percentile(corrs, perc_threshold)
idxs = (corrs > perc)
corrs_thresh = deepcopy(corrs)
corrs_thresh[~idxs] = np.nan

# load decoding coefs
model = pkl.load(open(coef_fname, 'rb'))
coefs_learned = model.best_estimator_.coef_.squeeze()
assert coefs_learned.size == idxs.sum(), 'Learned coefs should match size of repr'
coefs = deepcopy(corrs)
coefs[idxs] = coefs_learned
coefs[~idxs] = np.nan# !ls /home/chansingh/mntv1/deep-fMRI/results/linear_models/oct25

# Actual visualizations
# Note: for this to work, need to point the cortex config filestore to the `ds003020/derivative/pycortex-db` directory.
# This might look something like `/home/chansingh/mntv1/deep-fMRI/data/ds003020/derivative/pycortex-db/UTS03/anatomicals/`
def quickshow(X: np.ndarray, subject='UTS03', fname_save=None):
    vol = cortex.Volume(X, 'UTS03', xfmname='UTS03_auto')
    cortex.quickshow(vol, with_rois=True, cmap='PuBu') #, with_curvature=True, with_sulci=True)
    if fname_save is not None:
        plt.savefig(fname_save)
        plt.close()

print('saving flatmap corrs...')
quickshow(corrs, fname_save='../figs/flatmap_corrs.pdf')
# quickshow(corrs_thresh, fname_save='../figs/flatmap_corrs_thresh.pdf')

print('saving flatmap coefs...')
quickshow(coefs, fname_save='../figs/flatmap_coefs.pdf')