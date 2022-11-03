from os.path import join
import os
import pickle as pkl
import numpy as np
import sys
from sklearn.decomposition import PCA, NMF, FastICA
path_to_file = os.path.dirname(os.path.abspath(__file__))
sys.path.append(join(path_to_file, '..'))
import feature_spaces
import encoding_utils
viz_cortex = __import__('03_viz_cortex')
from tqdm import tqdm
import matplotlib.pyplot as plt

subject = 'UTS03'
out_dir = join(feature_spaces.data_dir, 'fmri_resp_norms', subject)

def calc_PCs(out_dir):
    train_stories, test_stories, allstories = \
        encoding_utils.get_allstories([1, 2, 3, 4, 5])

    print('loading responses...')
    zRresp = encoding_utils.get_response(
        train_stories, subject)  # shape (9461, 95556)

    print('calculating mean/std...')
    means = np.mean(zRresp, axis=0)
    stds = np.std(zRresp, axis=0)

    os.makedirs(out_dir, exist_ok=True)
    pkl.dump({'means': means, 'stds': stds}, open(
        join(out_dir, 'resps_means_stds.pkl'), 'wb'))

    print('fitting PCA...')
    pca = PCA().fit(zRresp)
    pkl.dump({'pca': pca}, open(
        join(out_dir, 'resps_pca.pkl'), 'wb'))

    print('fitting ICA...')
    ica = FastICA().fit(zRresp)
    pkl.dump({'ica': ica}, open(
        join(out_dir, 'resps_ica.pkl'), 'wb')) 

    print('fitting NMF...')
    try:
        nmf = NMF().fit(zRresp - zRresp.min())
        pkl.dump({'nmf': nmf}, open(
            join(out_dir, 'resps_nmf.pkl'), 'wb'))
    except:
        print('failed nmf!')

def viz_PCs(out_dir):
    os.makedirs(pc_dir, exist_ok=True)
    for k in ['pca', 'ica', 'nmf']:
        decomp = pkl.load(open(join(out_dir, 'resps_pca.pkl'), 'rb'))
        pc_dir = 'pcs_train'
        for i in tqdm(range(10)):
            viz_cortex.quickshow(decomp[k].components_[i]) # (n_components, n_features)
            plt.savefig(join(pc_dir, f'{k}_component_{i}.pdf'))
            plt.savefig(join(pc_dir, f'{k}_component_{i}.png'))
            plt.close()

if __name__ == '__main__':
    calc_PCs(out_dir)
    viz_PCs(out_dir)

