import sys
import numpy as np
from os.path import join
import h5py
import pickle as pkl
from copy import deepcopy
import matplotlib.pyplot as plt
import os.path
import cortex
path_to_file = os.path.dirname(os.path.abspath(__file__))
sys.path.append(join(path_to_file, '..'))
fit_decoding = __import__('02_fit_decoding')


def load_flatmap_data(
    encoding_result_dir='/home/chansingh/mntv1/deep-fMRI/results/encoding/bert-10__ndel=4/UTS03/',
    decoding_result_dir='/home/chansingh/mntv1/deep-fMRI/results/linear_models/oct25',
    decoding_result_fname='coef_rotten_tomatoes_bert-10__ndel=4fmri_perc=0_seed=1.pkl',
    calc_norms=True,
):
    # decoding_fname = 'coef_rotten_tomatoes_bert-10__ndel=4fmri_perc=0_seed=1.pkl',

    # !ls /home/chansingh/mntv1/deep-fMRI/results/linear_models/oct25
    # coef_fname = join(decoding_result, decoding_fname)

    # load test corrs for all voxels
    corrs = np.load(join(encoding_result_dir, 'corrs.npz'))['arr_0']
    reg_params = np.load(join(encoding_result_dir, 'valphas.npz'))['arr_0']

    # load / reshape / norm encoding weights
    weights = np.load(join(encoding_result_dir, 'weights.npz'))['arr_0']
    # pretty sure this is right, but might be switched...
    s = decoding_result_fname
    ndelays = int(s[s.index('ndel=') + len('ndel='): s.index('_fmri')])
    weights = weights.reshape(ndelays, -1, corrs.size)
    weights = weights.mean(axis=0).squeeze()  # mean over delays dimension...
    weights = np.linalg.norm(weights, axis=0)

    # find test corrs above a certain percentile
    # this is what was used to train the decoding model
    perc_threshold = int(
        decoding_result_fname[decoding_result_fname.index(
            'perc=') + len('perc='):]
        .split('_')[0])
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

    if not calc_norms:
        return {
            'encoding_weight_norms': weights,
            'corrs_thresh': corrs_thresh,
            'coefs': coefs,
            'reg_params': reg_params,
        }

    # load feature norms (recalculate if they don't exist)
    args = pkl.load(open(decoding_result.replace(
        'coef_', ''), 'rb')).reset_index()
    args = args.iloc[0]
    assert args.subject == 'UTS03', 'If changing the subject, need to change the encoding_result_dir'
    norms_file = join(encoding_result_dir,
                      decoding_result_fname
                      .replace('coef_', 'norms_')
                      # just save norms for everything
                      .replace(f'perc={perc_threshold}', 'perc=0')
                      )
    print('loading', norms_file)

    if os.path.exists(norms_file):
        norms = pkl.load(open(norms_file, 'rb'))
    else:
        X_train, y_train, X_test, y_test = fit_decoding.data.get_dsets(
            args.dset, seed=args.seed, subsample_frac=-1)

        feats_train, feats_test = fit_decoding.get_feats(
            args.model, X_train, X_test,
            subject_fmri=args.subject,
            perc_threshold_fmri=-1, args=args
        )

        norms = {
            'feats_train_mean': feats_train.mean(axis=0),
            'feats_train_std': feats_train.std(axis=0),
            'feats_test_mean': feats_test.mean(axis=0),
            'feats_test_std': feats_test.std(axis=0),
        }
        pkl.dump(norms, open(norms_file, 'wb'))

    return {
        'encoding_weight_norms': weights,
        'corrs_thresh': corrs_thresh,
        'coefs': coefs,
        'reg_params': reg_params,
        'contributions_test': coefs * norms['feats_test_mean'],
        'contributions_train': coefs * norms['feats_train_mean'],
        **norms,
    }


def quickshow(X: np.ndarray, subject='UTS03', fname_save=None):
    """
    Actual visualizations
    Note: for this to work, need to point the cortex config filestore to the `ds003020/derivative/pycortex-db` directory.
    Read the docs to find the config filestore file (smth like /home/chansingh/.config/pycortex/options.cfg)
    This might look something like `/home/chansingh/mntv1/deep-fMRI/data/ds003020/derivative/pycortex-db/UTS03/anatomicals/`
    """
    vol = cortex.Volume(X, subject, xfmname=f'{subject}_auto')
    # , with_curvature=True, with_sulci=True)
    cortex.quickshow(vol, with_rois=True, cmap='PuBu')
    if fname_save is not None:
        plt.savefig(fname_save)
        plt.savefig(fname_save.replace('.pdf', '.png'))
        plt.close()


if __name__ == '__main__':
    # decoding_result_dir = '/home/chansingh/mntv1/deep-fMRI/results/linear_models/oct25',
    # decoding_result_dir = '/home/chansingh/mntv1/deep-fMRI/results/linear_models/oct26_relu_and_normalization'
    # flatmaps = load_flatmap_data(
    #     decoding_result_dir=decoding_result_dir,
    #     # decoding_result_fname='coef_rotten_tomatoes_bert-10__ndel=4fmri_perc=0_seed=1.pkl',
    #     decoding_result_fname='coef_moral_stories_bert-10__ndel=4fmri_perc=0_seed=1.pkl',
    # )

    # decoding_result_dir = '/home/chansingh/mntv1/deep-fMRI/results/linear_models/probing_oct25_sub_0.05'
    # decoding_result_dir = '/home/chansingh/mntv1/deep-fMRI/results/linear_model/oct28_compare_joint'
    decoding_result_dir = '/home/chansingh/mntv1/deep-fMRI/results/linear_models/nov1_compare_joint'

    flatmaps = load_flatmap_data(
        decoding_result_dir=decoding_result_dir,
        # decoding_result_fname='coef_rotten_tomatoes_bert-10__ndel=4fmri_perc=0_seed=1.pkl',
        decoding_result_fname='coef_sst2_bert-10__ndel=4_fmri_perc=0_seed=1.pkl',
        # decoding_result_fname='coef_sst2_bert-10__ndel=4_fmri_perc=99_seed=1.pkl',
        calc_norms=True,
    )

    dict_to_save = {
        # decoding stuff
        '../figs/flatmaps/coefs.pdf': 'coefs',
        '../figs/flatmaps/contributions_test.pdf': 'contributions_test',
        # '../figs/flatmaps/contributions_train.pdf': 'contributions_train',

        # # encoding stuff
        # '../figs/flatmaps/reg_params.pdf': 'reg_params',
        # '../figs/flatmaps/encoding_weight_norms.pdf': 'encoding_weight_norms',
        # '../figs/flatmaps/corrs.pdf': 'corrs_thresh',

        # # data stuff
        '../figs/flatmaps/means_test.pdf': 'feats_test_mean',
        '../figs/flatmaps/stds_test.pdf': 'feats_test_std',
    }

    for k in dict_to_save:
        print('saving', k)
        quickshow(flatmaps[dict_to_save[k]], fname_save=join(path_to_file, k))
