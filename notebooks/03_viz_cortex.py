import numpy as np
from os.path import join
import h5py
import pickle as pkl
from copy import deepcopy
import matplotlib.pyplot as plt
import os.path
path_to_file = os.path.dirname(os.path.abspath(__file__))
import sys
sys.path.append(join(path_to_file, '..'))
fit_decoding = __import__('02_fit_decoding')


def load_corrs_and_coefs(
    encoding_result_dir='/home/chansingh/mntv1/deep-fMRI/results/encoding/bert-10__ndel=4/UTS03/',
    decoding_result_dir='/home/chansingh/mntv1/deep-fMRI/results/linear_models/oct25',
    decoding_result_fname='coef_rotten_tomatoes_bert-10__ndel=4fmri_perc=0_seed=1.pkl',
):
    # decoding_fname = 'coef_rotten_tomatoes_bert-10__ndel=4fmri_perc=0_seed=1.pkl',

    # !ls /home/chansingh/mntv1/deep-fMRI/results/linear_models/oct25
    # coef_fname = join(decoding_result, decoding_fname)

    # load test corrs for all voxels
    corrs = np.load(join(encoding_result_dir, 'corrs.npz'))['arr_0']

    # find test corrs above a certain percentile
    # this is what was used to train the decoding model
    perc_threshold = 0
    perc = np.percentile(corrs, perc_threshold)
    idxs = (corrs > perc)
    corrs_thresh = deepcopy(corrs)
    corrs_thresh[~idxs] = np.nan

    # load decoding coefs
    decoding_result = join(decoding_result_dir, decoding_result_fname)
    model = pkl.load(open(decoding_result, 'rb'))
    coefs_learned = model.best_estimator_.coef_.squeeze()
    assert coefs_learned.size == idxs.sum(), 'Learned coefs should match size of repr'
    coefs = deepcopy(corrs)
    coefs[idxs] = coefs_learned
    coefs[~idxs] = np.nan

    # load feature norms (recalculate if they don't exist)    
    args = pkl.load(open(decoding_result.replace('coef_', ''), 'rb')).reset_index()
    args = args.iloc[0]
    assert args.perc_threshold_fmri == 0, 'Should run this script with perc=0!'
    assert args.subject == 'UTS03', 'If changing the subject, need to change the encoding_result_dir'
    norms_file = join(encoding_result_dir, decoding_result_fname.replace('coef_', 'norms_'))
    if os.path.exists(norms_file):
        norms = pkl.load(open(norms_file, 'rb'))
    else:
        X_train, y_train, X_test, y_test = fit_decoding.data.get_dsets(
                args.dset, seed=args.seed, subsample_frac=0)

        feats_train, feats_test = fit_decoding.get_feats(
                args.model, X_train, X_test,
                subject_fmri=args.subject, perc_threshold_fmri=args.perc_threshold_fmri, args=args)

        norms = {
                'feats_train_mean': feats_train.mean(axis=0),
                'feats_train_std': feats_train.std(axis=0),
                'feats_test_mean': feats_test.mean(axis=0),
                'feats_test_std': feats_test.std(axis=0),
        }
        pkl.dump(norms, open(norms_file, 'wb'))
    return corrs_thresh, coefs, norms


if __name__ == '__main__':
    import cortex
    corrs_thresh, coefs, norms = load_corrs_and_coefs()

    def quickshow(X: np.ndarray, subject='UTS03', fname_save=None):
        """
        Actual visualizations
        Note: for this to work, need to point the cortex config filestore to the `ds003020/derivative/pycortex-db` directory.
        This might look something like `/home/chansingh/mntv1/deep-fMRI/data/ds003020/derivative/pycortex-db/UTS03/anatomicals/`
        """
        vol = cortex.Volume(X, 'UTS03', xfmname='UTS03_auto')
        # , with_curvature=True, with_sulci=True)
        cortex.quickshow(vol, with_rois=True, cmap='PuBu')
        if fname_save is not None:
            plt.savefig(fname_save)
            plt.close()

    print('saving flatmap corrs...')
    quickshow(corrs_thresh, fname_save=join(
        path_to_file, '../figs/flatmap_corrs.pdf'))
    # quickshow(corrs_thresh, fname_save='../figs/flatmap_corrs_thresh.pdf')

    print('saving flatmap coefs...')
    quickshow(coefs, fname_save=join(
        path_to_file, '../figs/flatmap_coefs.pdf'))
