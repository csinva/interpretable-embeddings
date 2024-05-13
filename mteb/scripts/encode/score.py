import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json
import time
import pandas as pd
from scipy import sparse
from bm25_pt import BM25
import pickle
import hashlib
import os
from transformers import AutoTokenizer

query_embedding_file = "./filtered_embeddings_dev_out_queries.json"
corpus_embedding_file = "./filtered_embeddings_dev_out_corpus.json"
query_file = "./runs/mixtral-dev-all/sample_dev_queries_all_beir.json"
corpus_file = "./runs/mixtral-dev-all/sample_dev_corpus_all_beir.json"

# Set limit for the number of entries read from all files
limit_query = None
limit_corpus = None
fit_option = "limited"  # Options: "full" for full corpus and queries, "limited" for limited corpus and queries

def load_json_file(file_path, limit=None):
    with open(file_path, "r") as f:
        data = json.load(f)
    if limit is not None:
        data = data[:limit]
    return data

def calculate_percentage(correct_matches, total_queries):
    return (correct_matches / total_queries) * 100

def process_queries(queries, queries_data, X, tsv_file, vectorizer, embeddings=None, bm25=None, use_query_embeddings=False):
    correct_matches = 0
    query_results = {}
    bm25_query_results = {}
    bm25_correct_matches = 0
    for i, query in enumerate(queries):
        if use_query_embeddings and embeddings is not None:
            query_id = queries_data[i]["query_id"]
            embedding = next(
                item for item in embeddings if item["query_id"] == query_id
            )["embedding"]
            norm = np.linalg.norm(embedding)
            query_vec = sparse.csr_matrix(embedding).astype(np.float64)
            if norm != 0:
                query_vec /= norm
        else:
            query_vec = vectorizer.transform([query])
            if embeddings is not None:
                query_id = queries_data[i]["query_id"]
                embedding = next(
                    item for item in embeddings if item["query_id"] == query_id
                )["embedding"]
                norm = np.linalg.norm(embedding)
                embedding = np.array(embedding).astype(np.float64)
                if norm != 0:
                    embedding /= norm
                query_vec = sparse.hstack([query_vec, sparse.csr_matrix(embedding)], format="csr")
        scores = cosine_similarity(X, query_vec)
        sorted_scores_indices = np.argsort(scores, axis=0)[::-1]
        top_10_scores_indices = sorted_scores_indices[:10]
        top_10_scores = [score[0][0] for score in scores[top_10_scores_indices].tolist()]  # Flatten the list
        top_10_corpus_ids = [corpus_data[idx]["corpus_id"] for idx in top_10_scores_indices.flatten()]
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
                        idx = next((i for i, item in enumerate(corpus_data) if item["corpus_id"] == cid), None)
                        correct_scores.append(scores[idx])
                        break
            correct_scores_str = ", ".join(map(str, correct_scores))
            # print(f"{query_id}, Incorrect: {max_score_corpus_id}, Correct: {correct_ids_str}, Score: {scores[max_score_index]}, Correct Scores: {correct_scores_str}")
        else:
            correct_matches += 1
        query_results[query_id] = {
            "top_10_scores": top_10_scores,
            "top_10_corpus_ids": top_10_corpus_ids,
            "max_score": scores[max_score_index][0],
            "correct": max_score_corpus_id in correct_corpus_ids
        }
        
        # BM25 scoring
        if bm25 is not None:
            bm25_scores = bm25.score(query)
            bm25_scores = bm25_scores.cpu().numpy()  # Convert the tensor to a NumPy array
            bm25_sorted_scores_indices = np.argsort(bm25_scores)[::-1]
            bm25_top_10_scores_indices = bm25_sorted_scores_indices[:10]
            bm25_top_10_scores = [bm25_scores[idx] for idx in bm25_top_10_scores_indices]
            bm25_top_10_corpus_ids = [corpus_data[idx]["corpus_id"] for idx in bm25_top_10_scores_indices]
            bm25_max_score_index = np.argmax(bm25_scores)
            bm25_max_score_corpus_id = corpus_data[bm25_max_score_index]["corpus_id"]
            if bm25_max_score_corpus_id in correct_corpus_ids:
                bm25_correct_matches += 1
            bm25_top_10_scores = [float(score) for score in bm25_top_10_scores]
            bm25_query_results[query_id] = {
                "top_10_scores": bm25_top_10_scores,
                "top_10_corpus_ids": bm25_top_10_corpus_ids,
                "max_score": float(bm25_scores[bm25_max_score_index]),
                "correct": bm25_max_score_corpus_id in correct_corpus_ids
            }
    return correct_matches, query_results, bm25_correct_matches, bm25_query_results


# Load the tokenizer for 'bert-large-cased'
tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased")
# Initialize BM25 with the tokenizer
bm25 = BM25(tokenizer=tokenizer, device='cuda')

# Load embeddings for corpus and queries
corpus_embeddings = load_json_file(corpus_embedding_file)
queries_embeddings = load_json_file(query_embedding_file)

# Load queries from json file
queries_data = load_json_file(query_file, limit_query)
queries = [item["query"] for item in queries_data]

# Load corpus from json file
corpus_data = load_json_file(corpus_file, limit_corpus)
corpus = [item["corpus"] for item in corpus_data]

if fit_option == "full":
    # Load the full queries
    print("fitting vectorizer on full data")
    with open('../../questions/msmarco/msmarco-dataset/queries.jsonl', 'r') as file:
        full_queries = [json.loads(line)['text'] for line in file]

    # Load the full corpus
    with open('../../questions/msmarco/msmarco-dataset/corpus.jsonl', 'r') as file:
        full_corpus = [json.loads(line)['text'] for line in file]

    # Fit the vectorizer on the full corpus and queries
    data_to_fit = full_corpus + full_queries
elif fit_option == "limited":
    # Fit the vectorizer on the limited corpus and queries
    data_to_fit = corpus + queries
else:
    raise ValueError("Invalid fit_option. Choose 'full' or 'limited'.")

# Fit the vectorizer

vectorizer_path = "./vectorizer.pkl"
data_hash_path = "./data_hash.txt"

# Function to create a hash of the data
def hash_data(data):
    return hashlib.md5(json.dumps(data, sort_keys=True).encode('utf-8')).hexdigest()

# Calculate the current data hash
current_data_hash = hash_data(data_to_fit)

if fit_option == "full":
    start_time = time.time()
    # Check if the vectorizer and data hash already exist
    if os.path.exists(vectorizer_path) and os.path.exists(data_hash_path):
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
    end_time = time.time()
    print(f"Vectorizer fitted in {end_time - start_time:.2f} seconds.")
else:
    vectorizer = TfidfVectorizer().fit(data_to_fit)

# Transform the corpus using the fitted vectorizer
corpus_vectors = vectorizer.transform(corpus)

embeddings_dict = {
    item["corpus_id"]: np.array(item["embedding"]) for item in corpus_embeddings
}

for key in embeddings_dict:
    norm = np.linalg.norm(embeddings_dict[key])
    if norm != 0:
        embeddings_dict[key] = embeddings_dict[key] / norm

sparse_embeddings = [
    sparse.csr_matrix(embeddings_dict[item["corpus_id"]]) for item in corpus_data
]

embeddings_sparse_matrix = sparse.vstack(sparse_embeddings)

X = sparse.hstack([corpus_vectors, embeddings_sparse_matrix], format="csr")

tsv_file = pd.read_csv(
    "../../questions/msmarco/msmarco-dataset/qrels/dev.tsv",
    sep="\t",
    header=0,
)

tsv_file = tsv_file.drop(tsv_file.columns[-1], axis=1)

# Initialize BM25
bm25.index(corpus)

# 'scores' now contains BM25 scores for the query relative to each document in the corpus.
# Initialize a dictionary to hold the percentage results
percentage_results = {}

print("With Embeddings:")
correct_matches_with_embeddings, query_results_with_embeddings, _, _ = process_queries(queries, queries_data, X, tsv_file, vectorizer, queries_embeddings, use_query_embeddings=False)
correct_matches_percentage_with_embeddings = calculate_percentage(correct_matches_with_embeddings, len(queries))
print(f"Percentage of correct matches: {correct_matches_percentage_with_embeddings}%\n")
percentage_results["With Embeddings"] = correct_matches_percentage_with_embeddings

print("Without Embeddings:")
X = corpus_vectors
correct_matches_without_embeddings, query_results_without_embeddings, _, _ = process_queries(queries, queries_data, X, tsv_file, vectorizer, use_query_embeddings=False)
correct_matches_percentage_without_embeddings = calculate_percentage(correct_matches_without_embeddings, len(queries))
print(f"Percentage of correct matches: {correct_matches_percentage_without_embeddings}%\n")
percentage_results["Without Embeddings"] = correct_matches_percentage_without_embeddings

print("With BM25:")
_, _, bm25_correct_matches, bm25_query_results = process_queries(queries, queries_data, X, tsv_file, vectorizer, bm25=bm25, use_query_embeddings=False)
bm25_correct_matches_percentage = calculate_percentage(bm25_correct_matches, len(queries))
print(f"Percentage of correct matches: {bm25_correct_matches_percentage}%\n")
percentage_results["With BM25"] = bm25_correct_matches_percentage

print("With Only Embeddings:")
X = embeddings_sparse_matrix
correct_matches_only_embeddings, query_results_only_embeddings, _, _ = process_queries(queries, queries_data, X, tsv_file, vectorizer, queries_embeddings, use_query_embeddings=True)
correct_matches_percentage_only_embeddings = calculate_percentage(correct_matches_only_embeddings, len(queries))
print(f"Percentage of correct matches: {correct_matches_percentage_only_embeddings}%\n")
percentage_results["With Only Embeddings"] = correct_matches_percentage_only_embeddings

# Ensure the score directory exists
score_directory = 'score'
os.makedirs(score_directory, exist_ok=True)

# Write the difference in max scores for queries that are classified correctly by both with and without embeddings to a file
with open(os.path.join(score_directory, 'difference_in_max_scores.txt'), 'w') as f:
    f.write("Difference in max scores for TF-IDF:\n")
    f.write(f"{'Query ID':<10}{'Score with embeddings':<25}{'Score without embeddings':<25}{'Difference':<10}\n")
    for query_id, result in query_results_with_embeddings.items():
        if result['correct'] and query_results_without_embeddings[query_id]['correct']:
            score_with_embeddings = result['max_score']
            score_without_embeddings = query_results_without_embeddings[query_id]['max_score']
            diff = score_with_embeddings - score_without_embeddings
            f.write(f"{query_id:<10}{score_with_embeddings:<25}{score_without_embeddings:<25}{diff:<10}\n")

# Save the query results to json files within the score folder
with open(os.path.join(score_directory, 'query_results_with_embeddings.json'), 'w') as f:
    json.dump(query_results_with_embeddings, f, indent=4)

with open(os.path.join(score_directory, 'query_results_without_embeddings.json'), 'w') as f:
    json.dump(query_results_without_embeddings, f, indent=4)

with open(os.path.join(score_directory, 'query_results_with_bm25.json'), 'w') as f:
    json.dump(bm25_query_results, f, indent=4)

with open(os.path.join(score_directory, 'query_results_only_embeddings.json'), 'w') as f:
    json.dump(query_results_only_embeddings, f, indent=4)

# Save the percentage results to a json file within the score folder
with open(os.path.join(score_directory, 'score.json'), 'w') as f:
    json.dump(percentage_results, f, indent=4)