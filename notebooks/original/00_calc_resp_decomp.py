from sklearn.decomposition import PCA, NMF, FastICA, DictionaryLearning
import numpy as np
import pickle as pkl
import ridge_utils.feature_spaces as feature_spaces
import ridge_utils.encoding_utils as encoding_utils
from tqdm import tqdm
import matplotlib.pyplot as plt
import sys
from os.path import join
import os

path_to_file = os.path.dirname(os.path.abspath(__file__))

sys.path.append(join(path_to_file, '..'))


viz_cortex = __import__('03_viz_cortex')

subject = 'UTS03'
out_dir = join(feature_spaces.data_dir, 'fmri_resp_norms', subject)


def calc_decomp(out_dir):
    train_stories, test_stories, allstories = \
        encoding_utils.get_allstories([1, 2, 3, 4, 5])

    print('loading responses...')
    zRresp = encoding_utils.get_response(
        train_stories, subject)  # shape (9461, 95556)

    print('calculating mean/std...')
    means = np.mean(zRresp, axis=0)
    stds = np.std(zRresp, axis=0)

    os.makedirs(out_dir, exist_ok=True)
    out_file = join(out_dir, 'resps_means_stds.pkl')
    pkl.dump({'means': means, 'stds': stds}, open(out_file, 'wb'))

    print('fitting PCA...')
    out_file = join(out_dir, 'resps_pca.pkl')
    if not os.path.exists(out_file):
        pca = PCA().fit(zRresp)
        pkl.dump({'pca': pca}, open(out_file, 'wb'))

    print('fitting ICA...')
    out_file = join(out_dir, 'resps_ica.pkl')
    if not os.path.exists(out_file):
        ica = FastICA().fit(zRresp)
        pkl.dump({'ica': ica}, open(out_file, 'wb'))

    print('fitting NMF...')
    try:
        out_file = join(out_dir, 'resps_nmf.pkl')
        if not os.path.exists(out_file):
            nmf = NMF(n_components=1000).fit(zRresp - zRresp.min())
            pkl.dump({'nmf': nmf}, open(out_file, 'wb'))
    except:
        print('failed nmf!')

    print('fitting SC...')
    try:
        out_file = join(out_dir, 'resps_sc.pkl')
        if not os.path.exists(out_file):
            sc = DictionaryLearning(n_components=1000).fit(
                zRresp - zRresp.min())
            pkl.dump({'sc': sc}, open(out_file, 'wb'))
    except:
        print('failed sc!')


def viz_decomp(out_dir):
    decomp_dir = join(path_to_file, 'decomps')
    os.makedirs(decomp_dir, exist_ok=True)
    for k in ['nmf', 'ica', 'pca', 'nmf', 'sc']:
        print('visualizing', k)
        decomp = pkl.load(open(join(out_dir, f'resps_{k}.pkl'), 'rb'))
        for i in tqdm(range(10)):
            # (n_components, n_features)
            viz_cortex.quickshow(decomp[k].components_[i])
            plt.savefig(join(decomp_dir, f'{k}_component_{i}.pdf'))
            plt.savefig(join(decomp_dir, f'{k}_component_{i}.png'))
            plt.close()


if __name__ == '__main__':
    calc_decomp(out_dir)
    # viz_decomp(out_dir)
