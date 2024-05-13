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

REPO_PATH = expanduser("~/mnt_qa/instructor")
BASE_PATH = join(REPO_PATH, 'scripts', 'encode')


def load_json_file(file_path, limit=None):
    with open(file_path, "r") as f:
        data = json.load(f)
    if limit is not None:
        data = data[:limit]
    return data


def calculate_percentage(correct_matches, total_queries):
    return (correct_matches / total_queries) * 100


def train_test_split(data, test_size=0.2):
    split_index = int(len(data) * (1 - test_size))
    return data[:split_index], data[split_index:]


def hash_data(data):
    return hashlib.md5(json.dumps(data, sort_keys=True).encode('utf-8')).hexdigest()


def process_queries(
        queries, queries_data, corpus_embeddings, tsv_file, vectorizer,
        lr=None, append_coeff=None, indices=None, embeddings=None, use_only_query_embeddings=False):
    '''Compute cosine similarity

    '''
    correct_matches = 0
    query_results = {}
    for i, query in enumerate(queries):
        if use_only_query_embeddings and embeddings is not None:
            query_id = queries_data[i]["query_id"]
            embedding = next(
                item for item in embeddings if item["query_id"] == query_id
            )["embedding"]
            if indices is not None:
                embedding = [embedding[i] for i in indices]
            query_vec = sparse.csr_matrix(embedding).astype(np.float64)
            query_vec = normalize(query_vec, axis=1)
        else:
            query_vec = vectorizer.transform([query])
            if embeddings is not None:
                query_id = queries_data[i]["query_id"]
                embedding = next(
                    item for item in embeddings if item["query_id"] == query_id
                )["embedding"]
                if indices is not None:
                    embedding = [embedding[i] *
                                 append_coeff[i]*lr for i in indices]
                embedding = np.array(embedding).astype(np.float64)
                query_vec = sparse.hstack(
                    [query_vec, sparse.csr_matrix(embedding)], format="csr")

        scores = cosine_similarity(corpus_embeddings, query_vec)
        sorted_scores_indices = np.argsort(scores, axis=0)[::-1]
        top_10_scores_indices = sorted_scores_indices[:10]
        # Flatten the list
        top_10_scores = [score[0][0]
                         for score in scores[top_10_scores_indices].tolist()]
        top_10_corpus_ids = [corpus_data[idx]["corpus_id"]
                             for idx in top_10_scores_indices.flatten()]
        max_score_index = np.argmax(scores)
        max_score_corpus_id = corpus_data[max_score_index]["corpus_id"]
        query_id = queries_data[i]["query_id"]
        correct_corpus_ids = tsv_file[
            (tsv_file["query-id"] == query_id)
        ]["corpus-id"].tolist()
        if max_score_corpus_id not in correct_corpus_ids:
            correct_ids_str = ", ".join(map(str, correct_corpus_ids))
            correct_scores = []
            for cid in correct_corpus_ids:
                for item in corpus_data:
                    if item["corpus_id"] == cid:
                        idx = next((i for i, item in enumerate(
                            corpus_data) if item["corpus_id"] == cid), None)
                        correct_scores.append(scores[idx])
                        break
            correct_scores_str = ", ".join(map(str, correct_scores))
        else:
            correct_matches += 1
        query_results[query_id] = {
            "top_10_scores": top_10_scores,
            "top_10_corpus_ids": top_10_corpus_ids,
            "max_score": scores[max_score_index][0],
            "correct": max_score_corpus_id in correct_corpus_ids
        }
    return correct_matches, query_results


def prepare_data_to_fit(fit_option, corpus, queries):
    if fit_option == "full":
        print("fitting vectorizer on full data")
        with open(join(REPO_PATH, 'questions/msmarco/msmarco-dataset/queries.jsonl'), 'r') as file:
            full_queries = [json.loads(line)['text'] for line in file]
        with open(join(REPO_PATH, 'questions/msmarco/msmarco-dataset/corpus.jsonl'), 'r') as file:
            full_corpus = [json.loads(line)['text'] for line in file]
        return full_corpus + full_queries
    elif fit_option == "limited":
        return corpus + queries
    else:
        raise ValueError("Invalid fit_option. Choose 'full' or 'limited'.")


def fit_vectorizer(fit_option, data_to_fit,
                   vectorizer_path=join(BASE_PATH, "./vectorizer.pkl"),
                   data_hash_path=join(BASE_PATH, "./data_hash.txt")):
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


def greedy_search(queries_embeddings, queries_data_train, tsv_file, vectorizer, corpus_vectors, base_score):
    embedding_len = len(queries_embeddings[0]["embedding"])
    candidate_indices = list(range(embedding_len))
    greedy_embeddings = []
    lr = 0.05
    append_coeff = [1] * embedding_len
    while len(candidate_indices) > 0:
        for i in candidate_indices:
            current_index = i
            indices_append = greedy_embeddings + [i]
            embeddings_dict = {item["corpus_id"]: np.array(
                item["embedding"]) for item in corpus_embeddings}
            # extract the indices worth of embeddings and overwrite embeddings_dict.
            for key in embeddings_dict:
                embeddings_dict[key] = [embeddings_dict[key][index] *
                                        append_coeff[i]*lr for i, index in enumerate(indices_append)]
            sparse_embeddings = [sparse.csr_matrix(
                embeddings_dict[item["corpus_id"]]) for item in corpus_data]
            embeddings_sparse_matrix = sparse.vstack(sparse_embeddings)
            X = sparse.hstack(
                [corpus_vectors, embeddings_sparse_matrix], format="csr")
            correct_matches_with_embeddings, query_results_with_embeddings = process_queries(
                queries, queries_data_train, X, tsv_file, vectorizer, lr=lr, append_coeff=append_coeff, indices=indices_append, embeddings=queries_embeddings_train, use_only_query_embeddings=False)
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
            embeddings_dict = {item["corpus_id"]: np.array(
                item["embedding"]) for item in corpus_embeddings}
            while True:
                append_coeff[best_index] += 1
                # extract the indices worth of embeddings and overwrite embeddings_dict.
                for key in embeddings_dict:
                    embeddings_dict[key] = [embeddings_dict[key][index] *
                                            append_coeff[i]*lr for i, index in enumerate(greedy_embeddings)]
                sparse_embeddings = [sparse.csr_matrix(
                    embeddings_dict[item["corpus_id"]]) for item in corpus_data]
                embeddings_sparse_matrix = sparse.vstack(sparse_embeddings)
                X = sparse.hstack(
                    [corpus_vectors, embeddings_sparse_matrix], format="csr")
                correct_matches_with_embeddings, query_results_with_embeddings = process_queries(
                    queries, queries_data_train, X, tsv_file, vectorizer, lr=lr, append_coeff=append_coeff, indices=indices_append, embeddings=queries_embeddings_train, use_only_query_embeddings=False)
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


if __name__ == "__main__":
    query_embedding_filename = join(
        BASE_PATH, "filtered_embeddings_dev_out_queries.json")
    corpus_embedding_filename = join(
        BASE_PATH, "filtered_embeddings_dev_out_corpus.json")
    query_filename = join(
        BASE_PATH, "runs/mixtral-dev-all/sample_dev_queries_all_beir.json")
    corpus_filename = join(
        BASE_PATH, "./runs/mixtral-dev-all/sample_dev_corpus_all_beir.json")
    tsv_filename = join(
        REPO_PATH, "questions/msmarco/msmarco-dataset/qrels/dev.tsv")

    # Set limit for the number of entries read from all files
    limit_query = None
    limit_corpus = None
    # Options: "full" for full corpus and queries, "limited" for limited corpus and queries
    fit_option = "limited"

    # Load embeddings for corpus and queries
    print('loading queries and corpus embeddings...')

    # these are both lists (len 6980 and 7433)
    # each element of list is dict with two keys, e.g. {'corpus_id': 3863, 'embedding': [0, 1, ..., 0, 1]}
    # embedding is list of length 333
    queries_embeddings = load_json_file(query_embedding_filename)
    corpus_embeddings = load_json_file(corpus_embedding_filename)

    # Split the query embeddings into train and test sets
    queries_embeddings_train, queries_embeddings_test = train_test_split(
        queries_embeddings, test_size=0.2)

    # Load queries from json file
    queries_data = load_json_file(query_filename, limit_query)
    queries_data_train, queries_data_test = train_test_split(
        queries_data, test_size=0.2)
    queries = [item["query"] for item in queries_data_train]

    # Load corpus from json file
    corpus_data = load_json_file(corpus_filename, limit_corpus)
    corpus = [item["corpus"] for item in corpus_data]

    data_to_fit = prepare_data_to_fit(fit_option, corpus, queries)
    vectorizer, current_data_hash = fit_vectorizer(fit_option, data_to_fit)

    # Transform the corpus using the fitted vectorizer
    corpus_vectors = vectorizer.transform(corpus)
    embeddings_dict = {item["corpus_id"]: np.array(
        item["embedding"]) for item in corpus_embeddings}

    tsv_file = pd.read_csv(tsv_filename, sep="\t", header=0)
    tsv_file = tsv_file.drop(tsv_file.columns[-1], axis=1)

    percentage_results = {}

    print("Without Embeddings:")
    X = corpus_vectors
    correct_matches_without_embeddings, query_results_without_embeddings = process_queries(
        queries, queries_data, X, tsv_file, vectorizer, use_only_query_embeddings=False)
    correct_matches_percentage_without_embeddings = calculate_percentage(
        correct_matches_without_embeddings, len(queries))
    print(
        f"Percentage of correct matches: {correct_matches_percentage_without_embeddings}%\n")
    percentage_results["Without Embeddings"] = correct_matches_percentage_without_embeddings

    print("With Greedy Embeddings:")
    base_score = correct_matches_percentage_without_embeddings
    greedy_embeddings, base_score = greedy_search(
        queries_embeddings, queries_data_train, tsv_file, vectorizer, corpus_vectors, base_score)
    percentage_results["With Greedy Embeddings"] = base_score
    print(greedy_embeddings, base_score)
