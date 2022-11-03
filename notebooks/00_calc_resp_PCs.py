from os.path import join
import os
import pickle as pkl
import numpy as np
import sys
from sklearn.decomposition import PCA, NMF
path_to_file = os.path.dirname(os.path.abspath(__file__))
sys.path.append(join(path_to_file, '..'))
import feature_spaces
import encoding_utils
viz_cortex = __import__('03_viz_cortex')

train_stories, test_stories, allstories = encoding_utils.get_allstories([
                                                                        1, 2, 3, 4, 5])
subject = 'UTS03'

print('loading responses...')
zRresp = encoding_utils.get_response(
    train_stories, subject)  # shape (9461, 95556)

print('calculating mean/std...')
means = np.mean(zRresp, axis=0)
stds = np.std(zRresp, axis=0)
out_dir = join(feature_spaces.data_dir, 'fmri_resp_norms', subject)
os.makedirs(out_dir, exist_ok=True)
pkl.dump({'means': means, 'stds': stds}, open(
    join(out_dir, 'resps_means_stds.pkl'), 'wb'))

print('fitting PCA...')
pca = PCA().fit(zRresp)
pkl.dump({'pca': pca}, open(
    join(out_dir, 'resps_pca.pkl'), 'wb'))

print('fitting NMF...')
nmf = NMF().fit(zRresp)
pkl.dump({'nmf': nmf}, open(
    join(out_dir, 'resps_nmf.pkl'), 'wb'))
