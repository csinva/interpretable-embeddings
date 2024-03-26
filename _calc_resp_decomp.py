from sklearn.decomposition import PCA, NMF, FastICA, DictionaryLearning
import numpy as np
import pickle as pkl
import feature_spaces
import encoding_utils
from tqdm import tqdm
import matplotlib.pyplot as plt
import sys
from os.path import join
import os
import story_names
import joblib
path_to_file = os.path.dirname(os.path.abspath(__file__))


def cache_resps(cache_fmri_resps_dir='/home/chansingh/cache_fmri_resps'):
    for subject in ['UTS03', 'UTS02', 'UTS01']:
        print('caching', subject)
        train_stories = story_names.get_story_names(subject, 'train')
        zRresp = encoding_utils.get_response(
            train_stories, subject)  # shape (27449, 95556)
        joblib.dump(zRresp, join(cache_fmri_resps_dir, f'{subject}.pkl'))


def calc_decomp(out_dir, subject, subsample_input=None):
    print('loading responses...')
    zRresp = joblib.load(
        join('/home/chansingh/cache_fmri_resps', f'{subject}.pkl'))
    print('shape', zRresp.shape)
    if subsample_input:
        zRresp = zRresp[::subsample_input]
        print('shape after subsampling', zRresp.shape)

    print('calculating mean/std...')
    means = np.mean(zRresp, axis=0)
    stds = np.std(zRresp, axis=0)

    os.makedirs(out_dir, exist_ok=True)
    out_file = join(out_dir, 'resps_means_stds.pkl')
    pkl.dump({'means': means, 'stds': stds}, open(out_file, 'wb'))

    print('fitting PCA...')
    out_file = join(out_dir, 'resps_pca.pkl')
    # if not os.path.exists(out_file):
    pca = PCA().fit(zRresp)
    joblib.dump(pca, out_file)

    # print('fitting ICA...')
    # out_file = join(out_dir, 'resps_ica.pkl')
    # if not os.path.exists(out_file):
    #     ica = FastICA().fit(zRresp)
    #     pkl.dump({'ica': ica}, open(out_file, 'wb'))

    # print('fitting NMF...')
    # try:
    #     out_file = join(out_dir, 'resps_nmf.pkl')
    #     if not os.path.exists(out_file):
    #         nmf = NMF(n_components=1000).fit(zRresp - zRresp.min())
    #         pkl.dump({'nmf': nmf}, open(out_file, 'wb'))
    # except:
    #     print('failed nmf!')

    # print('fitting SC...')
    # try:
    #     out_file = join(out_dir, 'resps_sc.pkl')
    #     if not os.path.exists(out_file):
    #         sc = DictionaryLearning(n_components=1000).fit(
    #             zRresp - zRresp.min())
    #         pkl.dump({'sc': sc}, open(out_file, 'wb'))
    # except:
    #     print('failed sc!')


def viz_decomp(out_dir):
    # decomp_dir = join(path_to_file, 'decomps')
    # os.makedirs(decomp_dir, exist_ok=True)
    sys.path.append(join(path_to_file, '..'))
    # viz_cortex = __import__('03_viz_cortex')
    for k in ['pca', 'nmf', 'ica']:  # , 'sc']:
        print('visualizing', k)
        decomp = pkl.load(open(join(out_dir, f'resps_{k}.pkl'), 'rb'))
        for i in tqdm(range(10)):
            # (n_components, n_features)
            viz_cortex.quickshow(decomp[k].components_[i])
            plt.savefig(join(out_dir, f'{k}_component_{i}.pdf'))
            plt.savefig(join(out_dir, f'{k}_component_{i}.png'))
            plt.close()


if __name__ == '__main__':
    # cache_resps()
    for subject in ['UTS03', 'UTS01', 'UTS02']:
        print(subject)
        out_dir = join(feature_spaces.data_dir, 'fmri_resp_norms', subject)
        os.makedirs(out_dir, exist_ok=True)
        calc_decomp(out_dir, subject, subsample_input=2)
        # viz_decomp(out_dir)
