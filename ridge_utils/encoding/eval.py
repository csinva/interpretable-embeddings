
import logging
import numpy as np
import os
import random
import time


def nancorr(x, y):
    mask = ~np.isnan(x) & ~np.isnan(y)
    return np.corrcoef(x[mask], y[mask])[0, 1]


def evaluate_pc_model_on_each_voxel(
        args, stim, resp,
        model_params_to_save, pca, scaler):
    if args.encoding_model == 'ridge':
        weights_pc = model_params_to_save['weights_pc']
        preds_pc = stim @ weights_pc
        model_params_to_save['weights'] = weights_pc * \
            scaler.scale_ @ pca.components_
        model_params_to_save['bias'] = scaler.mean_ @ pca.components_ + pca.mean_
        # note: prediction = stim @ weights + bias
    preds_voxels = pca.inverse_transform(
        scaler.inverse_transform(preds_pc)
    )  # (n_trs x n_voxels)
    corrs = []
    for i in range(preds_voxels.shape[1]):
        corrs.append(nancorr(preds_voxels[:, i], resp[:, i]))
    corrs = np.array(corrs)
    corrs[np.isnan(corrs)] = 0
    return corrs


def add_summary_stats(r, verbose=True):
    for key in ['corrs_test', 'corrs_tune', 'corrs_tune_pc', 'corrs_test_pc']:
        if key in r:
            r[key + '_mean'] = np.nanmean(r[key])
            r[key + '_median'] = np.nanmedian(r[key])
            r[key + '_frac>0'] = np.nanmean(r[key] > 0)
            r[key + '_mean_top1_percentile'] = np.nanmean(
                np.sort(r[key])[-len(r[key]) // 100:])
            r[key + '_mean_top5_percentile'] = np.nanmean(
                np.sort(r[key])[-len(r[key]) // 20:])

            # add r2 stats
            r[key.replace('corrs', 'r2') +
              '_mean'] = np.nanmean(r[key] * np.abs(r[key]))
            r[key.replace('corrs', 'r2') +
              '_median'] = np.nanmedian(r[key] * np.abs(r[key]))

            if key == 'corrs_test' and verbose:
                logging.info(f"mean {key}: {r[key + '_mean']:.4f}")
                logging.info(f"median {key}: {r[key + '_median']:.4f}")
                logging.info(f"frac>0 {key}: {r[key + '_frac>0']:.4f}")
                logging.info(
                    f"mean top1 percentile {key}: {r[key + '_mean_top1_percentile']:.4f}")
                logging.info(
                    f"mean top5 percentile {key}: {r[key + '_mean_top5_percentile']:.4f}")

    return r
