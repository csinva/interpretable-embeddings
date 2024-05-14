import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
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
REPO_PATH = expanduser("~/mnt_qa/instructor")
BASE_PATH = join(REPO_PATH, 'scripts', 'encode')


def _load_json_file(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)
    return data


def evaluate_retrieval(embs_queries, embs_corpus, labels, corpus_ids):
    '''
    embs: nsamples x nfeatures
    '''
    similarities = cosine_similarity(embs_queries, embs_corpus)
    ranks_list = []
    for row, label in zip(similarities, labels):
        idxs_sorted = np.isin(corpus_ids[np.argsort(row)[::-1]], label)
        ranks = np.arange(len(idxs_sorted))[idxs_sorted]
        ranks_list.append(1 + min(ranks))
    ranks_list = np.array(ranks_list)
    mrr = np.mean(1 / ranks_list)
    top1_frac = np.mean(ranks_list == 1)
    return mrr, top1_frac


class MiniMarcoDataset(Dataset):
    def __init__(self):
        query_embedding_filename = join(
            BASE_PATH, "filtered_embeddings_dev_out_queries.json")
        corpus_embedding_filename = join(
            BASE_PATH, "filtered_embeddings_dev_out_corpus.json")
        query_filename = join(
            BASE_PATH, "runs/mixtral-dev-all/sample_dev_queries_all_beir.json")
        corpus_filename = join(
            BASE_PATH, "runs/mixtral-dev-all/sample_dev_corpus_all_beir.json")
        labels_filename = join(
            REPO_PATH, "questions/msmarco/msmarco-dataset/qrels/dev.tsv")

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
        assert set(self.query_ids) == set(self.labels_df.index)
        assert set(self.query_ids) == set(self.embs_tfidf_queries_df.index)
        assert set(self.corpus_ids) == set(self.embs_tfidf_corpus_df.index)
        assert set(self.corpus_ids) == set(
            np.concatenate(self.labels_df.values))

    def __len__(self):
        return len(self.df)

    def get_random_neg_corpus_id(self, query_id: str):
        corpus_id_random = np.random.choice(self.corpus_ids)
        while corpus_id_random in self.labels_df.loc[query_id]:
            corpus_id_random = np.random.choice(self.corpus_ids)
        return corpus_id_random

    def __getitem__(self, query_id: str):
        return self.embs_qa_queries_df.loc[query_id].values, self.embs_tfidf_queries_df.loc[query_id].values, self.labels_df.loc[query_id].values


class LinearMapping(nn.Module):
    def __init__(self, in_features, out_features):
        super(LinearMapping, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.linear.weight.data = torch.eye(in_features, out_features)
        # intialize to very small
        # self.linear.weight.data = 0.01 * torch.randn(in_features, out_features)

    def forward(self, x):
        return self.linear(x)


if __name__ == "__main__":

    batch_size = 10

    # Load embeddings for corpus and queries
    dset = MiniMarcoDataset()
    query_ids_train, query_ids_test = train_test_split(
        dset.query_ids, random_state=1, test_size=0.2)

    for i in range(0, len(query_ids_train), batch_size):
        # all embs should be nsamples x nfeatures
        embs_qa, embs_tfidf, labels = dset[query_ids_train[i:i+batch_size]]

        # compute cosine similarity between query and corpus embeddings
        print('shapes', embs_tfidf.shape, dset.embs_tfidf_corpus_df.values.shape)
        similarities = cosine_similarity(
            embs_tfidf, dset.embs_tfidf_corpus_df.values)

        # check rank of correct labels
        # first argsort each row of similarities
        # then, for each row, find the rank of the correct labels
        # finally, average over all rows
        ranks = np.array([np.where(np.argsort(row) == label)
                          for row, label in zip(similarities, labels)])
        print(ranks)
