import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
import json
import time
import pandas as pd
from scipy import sparse
import pickle
import hashlib
import os
import os.path
from os.path import join, expanduser
from torch.utils.data import Dataset


# REPO_PATH = expanduser("~/Desktop/instructor")
REPO_PATH = expanduser("~/mnt_qa/instructor")
BASE_PATH = join(REPO_PATH, 'scripts', 'encode')


def hash_data(data):
    return hashlib.md5(json.dumps(data, sort_keys=True).encode('utf-8')).hexdigest()


def compute_score(query_embeddings, queries_data, corpus_embeddings, tsv_file):
    '''Compute cosine similarity

    '''
    scores = cosine_similarity(corpus_embeddings, query_embeddings)
    sorted_scores_indices = np.argsort(-scores, axis=0)
    top_10_scores_indices = sorted_scores_indices[:10, :]
    top_10_scores = scores[top_10_scores_indices, np.arange(scores.shape[1])]
    top_10_corpus_ids = np.array([corpus_texts[idx]["corpus_id"]
                                 for idx in top_10_scores_indices.flatten()]).reshape(top_10_scores_indices.shape)

    max_score_indices = np.argmax(scores, axis=0)
    max_score_corpus_ids = [corpus_texts[idx]["corpus_id"]
                            for idx in max_score_indices]

    correct_matches = 0
    query_results = {}
    for i, query_data in enumerate(queries_data):
        query_id = query_data["query_id"]
        correct_corpus_ids = tsv_file[tsv_file["query-id"]
                                      == query_id]["corpus-id"].tolist()
        if i < len(max_score_corpus_ids) and max_score_corpus_ids[i] in correct_corpus_ids:
            correct_matches += 1
        query_results[query_id] = {
            "top_10_scores": top_10_scores[:, i].tolist() if i < top_10_scores.shape[1] else [],
            "top_10_corpus_ids": top_10_corpus_ids[:, i].tolist() if i < top_10_corpus_ids.shape[1] else [],
            "max_score": scores[max_score_indices[i], i] if i < len(max_score_indices) else None,
            "correct": max_score_corpus_ids[i] in correct_corpus_ids if i < len(max_score_corpus_ids) else False
        }
    return correct_matches, query_results


def prepare_data_to_fit(tfidf_data_source, corpus, queries):
    if tfidf_data_source == "full":
        print("fitting vectorizer on full data")
        with open(join(REPO_PATH, 'questions/msmarco/msmarco-dataset/queries.jsonl'), 'r') as file:
            full_queries = [json.loads(line)['text'] for line in file]
        with open(join(REPO_PATH, 'questions/msmarco/msmarco-dataset/corpus.jsonl'), 'r') as file:
            full_corpus = [json.loads(line)['text'] for line in file]
        return full_corpus + full_queries
    elif tfidf_data_source == "limited":
        return corpus + queries
    else:
        raise ValueError("Invalid fit_option. Choose 'full' or 'limited'.")


def fit_vectorizer(fit_option, data_to_fit,
                   vectorizer_path=join(BASE_PATH, "vectorizer.pkl"),
                   data_hash_path=join(BASE_PATH, "data_hash.txt")):
    current_data_hash = hash_data(data_to_fit)
    if fit_option == "full" and os.path.exists(vectorizer_path) and os.path.exists(data_hash_path):
        with open(data_hash_path, 'r') as file:
            saved_data_hash = file.read()
        if saved_data_hash == current_data_hash:
            with open(vectorizer_path, 'rb') as f:
                vectorizer = pickle.load(f)
            print("Vectorizer loaded from file.")
        else:
            vectorizer = TfidfVectorizer().fit(data_to_fit)
            with open(vectorizer_path, 'wb') as f:
                pickle.dump(vectorizer, f)
            with open(data_hash_path, 'w') as file:
                file.write(current_data_hash)
            print("Data has changed, vectorizer refitted and saved to file.")
    else:
        vectorizer = TfidfVectorizer().fit(data_to_fit)
        with open(vectorizer_path, 'wb') as f:
            pickle.dump(vectorizer, f)
        with open(data_hash_path, 'w') as file:
            file.write(current_data_hash)
        print("Vectorizer fitted and saved to file for the first time.")
    return vectorizer, current_data_hash


def greedy_search(queries, queries_embeddings, queries_data_train, tsv_file, corpus_vectors, base_score, vectorizer):
    embedding_len = len(queries_embeddings[0]["embedding"])
    candidate_indices = list(range(embedding_len))
    greedy_embeddings = []
    lr = 0.05
    append_coeff = [1] * embedding_len
    while len(candidate_indices) > 0:
        for i in candidate_indices:
            current_index = i
            indices_append = greedy_embeddings + [i]
            embs_qa_corpus_dict = {item["corpus_id"]: np.array(
                item["embedding"]) for item in embs_qa_corpus}
            # extract the indices worth of embeddings and overwrite embs_qa_corpus_dict.
            for key in embs_qa_corpus_dict:
                embs_qa_corpus_dict[key] = [embs_qa_corpus_dict[key][index] *
                                            append_coeff[i]*lr for i, index in enumerate(indices_append)]
            sparse_embeddings = [sparse.csr_matrix(
                embs_qa_corpus_dict[item["corpus_id"]]) for item in corpus_texts]
            embeddings_sparse_matrix = sparse.vstack(sparse_embeddings)
            X = sparse.hstack(
                [corpus_vectors, embeddings_sparse_matrix], format="csr")
            query_vectors = compute_query_embeddings(queries, queries_data_train, vectorizer, lr=lr, append_coeff=append_coeff,
                                                     indices=indices_append, embeddings=embs_qa_queries_train, use_only_query_embeddings=False)
            correct_matches_with_embeddings, query_results_with_embeddings = compute_score(
                query_vectors, queries_data_train, X, tsv_file)
            correct_matches_percentage_with_embeddings = calculate_percentage(
                correct_matches_with_embeddings, len(queries))
            if correct_matches_percentage_with_embeddings > base_score:
                base_score = correct_matches_percentage_with_embeddings
                best_index = current_index
            else:
                candidate_indices.remove(i)
        if best_index is not None:
            greedy_embeddings.append(best_index)
            # max out lr for the current best dimention.
            embs_qa_corpus_dict = {item["corpus_id"]: np.array(
                item["embedding"]) for item in embs_qa_corpus}
            while True:
                append_coeff[best_index] += 1
                # extract the indices worth of embeddings and overwrite embs_qa_corpus_dict.
                for key in embs_qa_corpus_dict:
                    embs_qa_corpus_dict[key] = [embs_qa_corpus_dict[key][index] *
                                                append_coeff[i]*lr for i, index in enumerate(greedy_embeddings)]
                sparse_embeddings = [sparse.csr_matrix(
                    embs_qa_corpus_dict[item["corpus_id"]]) for item in corpus_texts]
                embeddings_sparse_matrix = sparse.vstack(sparse_embeddings)
                X = sparse.hstack(
                    [corpus_vectors, embeddings_sparse_matrix], format="csr")
                query_vectors = compute_query_embeddings(queries, queries_data_train, vectorizer, lr=lr, append_coeff=append_coeff,
                                                         indices=indices_append, embeddings=embs_qa_queries_train, use_only_query_embeddings=False)
                correct_matches_with_embeddings, query_results_with_embeddings = compute_score(
                    query_vectors, queries_data_train, X, tsv_file)
                correct_matches_percentage_with_embeddings = calculate_percentage(
                    correct_matches_with_embeddings, len(queries))
                if correct_matches_percentage_with_embeddings > base_score:
                    print(correct_matches_percentage_with_embeddings, base_score,
                          indices_append, append_coeff, len(indices_append))
                    base_score = correct_matches_percentage_with_embeddings
                else:
                    append_coeff[best_index] -= 1
                    break
        best_index = None
    return greedy_embeddings, base_score


def compute_query_embeddings(queries, queries_data, vectorizer, lr=None, append_coeff=None, indices=None, embeddings=None, use_only_query_embeddings=False):
    query_ids = [qd["query_id"] for qd in queries_data]

    if use_only_query_embeddings and embeddings is not None:
        embs_qa_corpus_dict = {emb["query_id"]: emb["embedding"]
                               for emb in embeddings}
        embeddings_matrix = [embs_qa_corpus_dict[qid] for qid in query_ids]
        if indices is not None:
            embeddings_matrix = [[emb[i] for i in indices]
                                 for emb in embeddings_matrix]
        query_embeddings = sparse.csr_matrix(
            embeddings_matrix).astype(np.float64)
        query_embeddings = normalize(query_embeddings, axis=1)
    else:
        query_embeddings = vectorizer.transform(queries)
        if embeddings is not None:
            embs_qa_corpus_dict = {emb["query_id"]: emb["embedding"]
                                   for emb in embeddings}
            embeddings_matrix = [embs_qa_corpus_dict[qid] for qid in query_ids]
            if indices is not None:
                embeddings_matrix = [
                    [emb[i] * append_coeff[i] * lr for i in indices] for emb in embeddings_matrix]
            embeddings_matrix = sparse.csr_matrix(
                embeddings_matrix).astype(np.float64)
            query_embeddings = sparse.hstack(
                [query_embeddings, embeddings_matrix], format="csr")

    return query_embeddings


def _load_json_file(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)
    return data


def train_test_split_deterministic(data, test_size=0.2):
    split_index = int(len(data) * (1 - test_size))
    return data[:split_index], data[split_index:]


def calculate_percentage(correct_matches, total_queries):
    return (correct_matches / total_queries) * 100


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
        self.embs_qa_queries_df = pd.DataFrame(
            _load_json_file(query_embedding_filename)).set_index("query_id")
        self.embs_qa_corpus_df = pd.DataFrame(
            _load_json_file(corpus_embedding_filename)).set_index("corpus_id")

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

    def __getitem__(self, query_id: str):
        text = self.df.index[idx]
        inputs = self.tokenizer(text, padding="max_length",
                                truncation=True, max_length=512, return_tensors="pt")
        labels_onehot = torch.tensor(np.eye(2)[self.df.iloc[idx].values])
        return inputs.to(device), labels_onehot.to(device)


if __name__ == "__main__":
    m = MiniMarcoDataset()

    # Set limit for the number of entries read from all files
    limit_query = None
    limit_corpus = None
    # Options: "full" for full corpus and queries, "limited" for limited corpus and queries

    # Load embeddings for corpus and queries
    print('loading qa embeddings...')

    percentage_results = {}

    # print("Without QA Embeddings (tf-idf only):")
    # correct_matches_without_embeddings, query_results_without_embeddings = compute_score(
    #     query_vectors, queries_texts, corpus_vectors, labels_file)
    # correct_matches_percentage_without_embeddings = calculate_percentage(
    #     correct_matches_without_embeddings, len(queries))
    # print(
    #     f"Percentage of correct matches: {correct_matches_percentage_without_embeddings}%\n")
    # percentage_results["Without Embeddings"] = correct_matches_percentage_without_embeddings

    # print("With QA Embeddings:")
    # base_score = correct_matches_percentage_without_embeddings
    # greedy_embeddings, base_score = greedy_search(
    #     queries, embs_qa_queries, queries_texts_train, labels_file, corpus_vectors, base_score, vectorizer)
    # percentage_results["With Greedy Embeddings"] = base_score
    # print(greedy_embeddings, base_score)
