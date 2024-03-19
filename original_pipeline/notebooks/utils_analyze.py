import numpy as np
import sys
sys.path.append('..')
from os.path import join
from matplotlib import pyplot as plt
from collections import defaultdict
import pandas as pd
import os
import seaborn as sns
from tqdm import tqdm
import dvu
import viz
dvu.set_style()

def load_results(save_dir):
    dfs = []
    fnames = [
        fname for fname in os.listdir(save_dir)[::-1]
        if not fname.startswith('coef')
    ]
    for fname in tqdm(fnames):
        df = pd.read_pickle(join(save_dir, fname))
        # print(fname)
        # display(df)
        dfs.append(df.reset_index())
    d = pd.concat(dfs)
    # d = d.drop(columns='coef_')
    # .round(2)
    # d.set_index(['feats', 'dset'], inplace=True)
    d['nonlin_suffix'] = d['nonlinearity'].fillna('').str.replace('None', '').str.replace('tanh', '_tanh')
    d['model'] = d['model'] + d['nonlin_suffix']
    d['model_full'] = d['model'] + '_thresh=' + d['perc_threshold_fmri'].astype(str)
    return d