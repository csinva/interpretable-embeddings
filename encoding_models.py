import os
import sys
import datasets
import numpy as np
import json
from os.path import join, dirname
from functools import partial
from ridge_utils.data_sequence import DataSequence
from typing import Dict, List
from tqdm import tqdm
from ridge_utils.interp_data import lanczosinterp2D
from ridge_utils.semantic_model import SemanticModel
from transformers.pipelines.pt_utils import KeyDataset
from ridge_utils.utils_ds import apply_model_to_words, make_word_ds, make_phoneme_ds
from ridge_utils.utils_stim import load_textgrids, load_simulated_trfiles
from transformers import pipeline
import logging
import numpy as np
from sklearn.datasets import make_regression
from torch import nn


class MLP(nn.Module):
    def __init__(self, dim_inputs=768, dim_hidden=100, dim_outputs=2, nonlin=nn.ReLU()):
        super(MLP, self).__init__()

        self.dense0 = nn.Linear(dim_inputs, dim_hidden)
        self.nonlin = nonlin
        self.dense1 = nn.Linear(dim_hidden, dim_outputs)
        # self.output = nn.Linear(n_outputs, n_outputs)

    def forward(self, X, **kwargs):
        X = self.nonlin(self.dense0(X))
        # X = self.dropout(X)
        X = self.dense1(X)
        return X