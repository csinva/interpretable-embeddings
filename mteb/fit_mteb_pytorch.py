import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
import json
import pandas as pd
import os.path
from torch import nn
import torch.optim
from os.path import join, expanduser
from torch.utils.data import Dataset


# REPO_PATH = expanduser("~/Desktop/instructor")
# REPO_PATH = expanduser("~/mnt_qa/instructor")
# BASE_PATH = join(REPO_PATH, 'scripts', 'encode')
BASE_PATH = expanduser('~/interpretable-embeddings/data')


def _load_json_file(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)
    return data


def evaluate_retrieval(embs_queries, embs_corpus, labels, corpus_ids):
    '''
    embs: nsamples x nfeatures
    '''
    similarities = cosine_similarity(embs_queries, embs_corpus)
    # similarities = -1 * euclidean_distances(embs_queries, embs_corpus)
    ranks_list = []
    for row, label in zip(similarities, labels):
        idxs_sorted = np.isin(corpus_ids[np.argsort(row)[::-1]], label)
        ranks = np.arange(len(idxs_sorted))[idxs_sorted]

        # if len(ranks) == 0:
        #     ranks_list.append(len(corpus_ids))
        # else:
        ranks_list.append(1 + min(ranks))
    ranks_list = np.array(ranks_list)
    mrr = np.mean(1 / ranks_list)
    top1_frac = np.mean(ranks_list == 1)
    top3_frac = np.mean(ranks_list <= 3)

    # stds
    mrr_sem = np.std(1 / ranks_list) / np.sqrt(len(ranks_list))
    top1_frac_sem = np.std(ranks_list == 1) / np.sqrt(len(ranks_list))
    top3_frac_sem = np.std(ranks_list <= 3) / np.sqrt(len(ranks_list))

    return mrr, top1_frac, top3_frac, mrr_sem, top1_frac_sem, top3_frac_sem


class MiniMarcoDataset(Dataset):
    def __init__(self):
        # query_embedding_filename = join(
        #     BASE_PATH, "filtered_embeddings_dev_out_queries.json")
        # corpus_embedding_filename = join(
        #     BASE_PATH, "filtered_embeddings_dev_out_corpus.json")
        # query_filename = join(
        #     BASE_PATH, "runs/mixtral-dev-all/sample_dev_queries_all_beir.json")
        # corpus_filename = join(
        #     BASE_PATH, "runs/mixtral-dev-all/sample_dev_corpus_all_beir.json")
        # labels_filename = join(
        #     REPO_PATH, "questions/msmarco/msmarco-dataset/qrels/dev.tsv")
        query_embedding_filename = join(
            BASE_PATH, "queries_embeddings.json")
        corpus_embedding_filename = join(
            BASE_PATH, "corpus_embeddings.json")
        query_filename = join(
            BASE_PATH, "filtered_queries.json")
        corpus_filename = join(
            BASE_PATH, "filtered_corpus.json")
        labels_filename = join(
            BASE_PATH, "dev.tsv")

        # queries_df: 6980 rows, each value is an embedding, index is query_id
        # corpus_df: 7433 rows, each value is an embeddingm index is corpus_id
        # embedding is list of length 333
        print('loading qa embeddings...')
        self.embs_qa_queries_df = pd.DataFrame(
            _load_json_file(query_embedding_filename)).set_index("query_id")
        self.embs_qa_queries_df = pd.DataFrame(np.vstack(
            [x[0] for x in self.embs_qa_queries_df.values]), index=self.embs_qa_queries_df.index)
        self.embs_qa_corpus_df = pd.DataFrame(
            _load_json_file(corpus_embedding_filename)).set_index("corpus_id")
        self.embs_qa_corpus_df = pd.DataFrame(np.vstack(
            [x[0] for x in self.embs_qa_corpus_df.values]), index=self.embs_qa_corpus_df.index)

        print('computing tf-idf embeddings...')
        # df with actual underlying text (len 6980 and 7433)
        queries_text_df = pd.DataFrame(_load_json_file(
            query_filename))
        corpus_text_df = pd.DataFrame(_load_json_file(
            corpus_filename))
        texts_full = queries_text_df['query'].tolist(
        ) + corpus_text_df['corpus'].tolist()
        vectorizer = TfidfVectorizer().fit(texts_full)
        # vectorizer = CountVectorizer(ngram_range=(1, 3)).fit(texts_full)
        self.embs_tfidf_queries_df = pd.DataFrame(
            vectorizer.transform(queries_text_df['query']).toarray(),
            index=queries_text_df['query_id'])
        self.embs_tfidf_corpus_df = pd.DataFrame(
            vectorizer.transform(corpus_text_df['corpus']).toarray(),
            index=corpus_text_df['corpus_id'])

        # initially, 2 columns where rows give matching query-id/corpus-id matches(note: both query-ids and corpus-ids can be repeated
        # after groupby, 1 column where rows give list of corpus-ids for each query-id (mostly have length 1, but occasionally length 2, 3, or 4)
        self.labels_df = pd.read_csv(labels_filename, sep="\t", header=0)
        self.labels_df = self.labels_df.groupby(
            'query-id')['corpus-id'].agg(list)

        # check if all query_ids match
        self.query_ids = self.embs_qa_queries_df.index.unique()
        self.corpus_ids = self.embs_qa_corpus_df.index.unique()

        # remove anything where there isn't a valid corpus id
        self.labels_df = self.labels_df.apply(
            lambda l: [x for x in l if x in self.corpus_ids])
        self.query_ids = [k for k in self.query_ids if len(
            self.labels_df.loc[k]) > 0]

        for k in set(self.query_ids):
            assert k in self.labels_df.index

        # assert set(self.query_ids) == set(self.embs_tfidf_queries_df.index)
        # assert set(self.corpus_ids) == set(self.embs_tfidf_corpus_df.index)
        # assert set(self.corpus_ids) == set(
        #     np.concatenate(self.labels_df.values))

    def __len__(self):
        return len(self.query_ids)

    def get_random_neg_corpus_id(self, query_id: str):
        corpus_id_random = np.random.choice(self.corpus_ids)
        while corpus_id_random in self.labels_df.loc[query_id]:
            corpus_id_random = np.random.choice(self.corpus_ids)
        return corpus_id_random

    def __getitem__(self, query_id: str):
        return self.embs_qa_queries_df.loc[query_id].values, self.embs_tfidf_queries_df.loc[query_id].values, self.labels_df.loc[query_id].values
