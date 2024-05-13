"""
    ToDo:
    split test and train. 80 / 20 (for queries only).
    calculate score.
    one at a time, add every dimention.
    then finally for the best features, show improvement on test. (run a score for test for every dimention that you add). ideally the graph should go up. 

combined X (norm = /330)
69.4663323782235 69.0365329512894% [5, 29, 55, 57, 78, 83, 92, 112, 121, 205, 270, 300, 312, 321, 329] 15
combined X (norm = /15)
69.55587392550143 69.0365329512894% [1, 3, 5, 6, 19, 20, 22, 23, 24, 33, 115, 121, 167, 175, 176, 190, 206, 211, 212, 244, 271, 274, 295, 307, 310, 313] 26
combined X (norm = /10)
70.80945558739255 70.79154727793696 [3, 5, 6, 8, 10, 13, 18, 20, 22, 24, 32, 33, 35, 39, 42, 44, 49, 54, 61, 87, 96, 109, 115, 116, 117, 119, 121, 127, 128, 132, 133, 134, 137, 140, 147, 148, 153, 154, 155, 156, 158, 167, 168, 175, 176, 180, 185, 187, 203, 206, 207, 210, 211, 212, 218, 219, 221, 229, 231, 233, 237, 239, 240, 248, 249, 261, 271, 307] 68
combined X (norm = /5) (best)
73.28080229226362 73.26289398280802 [0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 13, 17, 24, 25, 32, 33, 36, 39, 42, 44, 45, 46, 47, 53, 56, 57, 59, 61, 71, 72, 75, 83, 85, 87, 88, 90, 93, 96, 101, 109, 116, 118, 119, 124, 125, 127, 128, 130, 131, 133, 139, 142, 147, 148, 151, 152, 153, 154, 157, 158, 168, 175, 180, 181, 182, 183, 184, 190, 192, 194, 195, 200, 201, 218, 219, 224, 226, 227, 229, 231, 237, 239, 240, 243, 249, 250, 251, 255, 259, 262, 270, 286, 287, 299, 304, 308, 309, 314, 315, 326, 331] 101
combined X (norm = /3)
73.11962750716332 69.0365329512894% [0, 2, 3, 5, 6, 7, 8, 10, 15, 16, 18, 20, 24, 25, 29, 30, 31, 32, 33, 36, 39, 40, 47, 53, 57, 61, 63, 72, 75, 78, 85, 90, 93, 95, 98, 101, 102, 103, 109, 114, 115, 118, 119, 121, 128, 130, 132, 135, 142, 147, 151, 153, 154, 158, 168, 172, 175, 180, 182, 183, 188, 194, 195, 196, 197, 198, 205, 208, 226, 233, 235, 237, 238, 240, 247, 251, 253, 260, 261, 266, 273, 276, 292, 304, 308, 314, 315, 316, 323, 332] 90
combined X (norm = /2)
71.59742120343839 71.57951289398281 [2, 3, 5, 8, 10, 16, 18, 30, 33, 40, 45, 48, 53, 57, 58, 61, 75, 78, 92, 99, 102, 109, 115, 118, 119, 121, 132, 136, 142, 143, 151, 164, 177, 182, 197, 198, 203, 208, 214, 230, 237, 243, 245, 249, 253, 255, 258, 270, 272, 273, 276, 292, 294, 295, 299, 300, 303, 305, 309, 316, 323, 332] 62
combined X (norm = /np.sqrt(len_embeddings))
70.8810888252149 70.82736389684814 [5, 10, 11, 12, 13, 14, 24, 25, 27, 28, 29, 30, 31, 32, 33, 34, 39, 44, 45, 101, 105, 127, 132, 133, 167, 168, 175, 176, 177, 180, 182, 183, 206, 219] 34
combined X (norm = /len_embeddings
bad
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
import json
import time
import pandas as pd
from scipy import sparse
from bm25_pt import BM25
import pickle
import hashlib
import os
from transformers import AutoTokenizer

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

def process_queries(queries, queries_data, X, tsv_file, vectorizer,  lr=None, append_coeff=None, indices=None, corpus_embeddings=None, embeddings=None, bm25=None, use_only_query_embeddings=False):
    correct_matches = 0
    query_results = {}
    bm25_query_results = {}
    bm25_correct_matches = 0
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
                    embedding = [embedding[i]*append_coeff[i]*lr for i in indices]
                embedding = np.array(embedding).astype(np.float64)
                query_vec = sparse.hstack([query_vec, sparse.csr_matrix(embedding)], format="csr")
                # query_vec = normalize_embedding_0(query_vec, indices, axis=1)

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

def normalize_embedding_0(X, indices, axis=1):
    """
    X: is a csr matrix or vector
    indices: the indices of the embeddings
    axis: axis to normalize across
    """
    len_embeddings = len(indices)
    # X_full =  normalize(X, axis=axis)
    #extract the tfidf embeddings and custom embeddings and normalize only the custom embeddings
    total_features = X.shape[1]
    n_corpus_features = total_features - len_embeddings
    corpus_vectors_retrieved = X[:, :n_corpus_features]
    embeddings_sparse_matrix_retrieved = X[:, n_corpus_features:]
    #ToDo: there is an issue here. as all values will be 1.0 even after normalization. You might wanna divide these with the total len of embeddings. 
    normalized_embeddings_sparse_matrix_retrieved = normalize(embeddings_sparse_matrix_retrieved)
    # normalized_embeddings_sparse_matrix_retrieved = embeddings_sparse_matrix_retrieved / 5
    combined_X = sparse.hstack([corpus_vectors_retrieved, normalized_embeddings_sparse_matrix_retrieved], format="csr")

    # vars_to_print = ['X_full', 'combined_X', 'normalized_embeddings_sparse_matrix_retrieved', 'embeddings_sparse_matrix_retrieved', 'corpus_vectors_retrieved']
    # for var in vars_to_print:
    #     print(f"{var}:")
    #     print((locals()[var]).shape)
    #     print("--------")
    # exit(0)
    # return X_full
    return combined_X

# Load the tokenizer for 'bert-large-cased'
tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased")
# Initialize BM25 with the tokenizer
# bm25 = BM25(tokenizer=tokenizer, device='cuda')
bm25 = BM25(tokenizer=tokenizer)


query_embedding_file = "./filtered_embeddings_dev_out_queries.json"
corpus_embedding_file = "./filtered_embeddings_dev_out_corpus.json"
query_file = "./runs/mixtral-dev-all/sample_dev_queries_all_beir.json"
corpus_file = "./runs/mixtral-dev-all/sample_dev_corpus_all_beir.json"
# Set limit for the number of entries read from all files
limit_query = None
limit_corpus = None
fit_option = "limited"  # Options: "full" for full corpus and queries, "limited" for limited corpus and queries


# Load embeddings for corpus and queries
queries_embeddings = load_json_file(query_embedding_file)
corpus_embeddings = load_json_file(corpus_embedding_file)
# Split the query embeddings into train and test sets
queries_embeddings_train, queries_embeddings_test = train_test_split(queries_embeddings, test_size=0.2)

# Load queries from json file
queries_data = load_json_file(query_file, limit_query)
queries_data_train, queries_data_test = train_test_split(queries_data, test_size=0.2)
queries = [item["query"] for item in queries_data_train]

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

embeddings_dict = {item["corpus_id"]: np.array(item["embedding"]) for item in corpus_embeddings}

tsv_file = pd.read_csv("../../questions/msmarco/msmarco-dataset/qrels/dev.tsv", sep="\t", header=0)
tsv_file = tsv_file.drop(tsv_file.columns[-1], axis=1)

# Initialize BM25
# bm25.index(corpus)

# 'scores' now contains BM25 scores for the query relative to each document in the corpus.
# Initialize a dictionary to hold the percentage results
percentage_results = {}

print("Without Embeddings:")
X = corpus_vectors
correct_matches_without_embeddings, query_results_without_embeddings, _, _ = process_queries(queries, queries_data, X, tsv_file, vectorizer, use_only_query_embeddings=False)
correct_matches_percentage_without_embeddings = calculate_percentage(correct_matches_without_embeddings, len(queries))
print(f"Percentage of correct matches: {correct_matches_percentage_without_embeddings}%\n")
percentage_results["Without Embeddings"] = correct_matches_percentage_without_embeddings

###########################

# average_cos_sim = avg([item["max_score"] for item in query_results_without_embeddings])
base_score = correct_matches_percentage_without_embeddings
# query_results[query_id] = {
#             "top_10_scores": top_10_scores,
#             "top_10_corpus_ids": top_10_corpus_ids,
#             "max_score": scores[max_score_index][0],
#             "correct": max_score_corpus_id in correct_corpus_ids
#         }
print("With Greedy Embeddings:")
# Iterate over length of custom embedding
embedding_len = len(queries_embeddings[0]["embedding"])

candidate_indices = list(range(embedding_len))
greedy_embeddings = []
lr = 0.05
append_coeff = [1] * embedding_len
while len(candidate_indices) > 0:
    for i in candidate_indices:
        current_index = i
        indices_append = greedy_embeddings + [i]
        embeddings_dict = {item["corpus_id"]: np.array(item["embedding"]) for item in corpus_embeddings}
        #extract the indices worth of embeddings and overwrite embeddings_dict.
        for key in embeddings_dict:
            embeddings_dict[key] = [embeddings_dict[key][index]*append_coeff[i]*lr for i, index in enumerate(indices_append)]
        sparse_embeddings = [sparse.csr_matrix(embeddings_dict[item["corpus_id"]]) for item in corpus_data]
        embeddings_sparse_matrix = sparse.vstack(sparse_embeddings)
        X = sparse.hstack([corpus_vectors, embeddings_sparse_matrix], format="csr")
        # X = normalize_embedding_0(X, indices_append, axis=1)
#Todo: send the LR array to the score function too to apply to corpus. 
        correct_matches_with_embeddings, query_results_with_embeddings, _, _ = process_queries(queries, queries_data_train, X, tsv_file, vectorizer, lr=lr, append_coeff=append_coeff, indices=indices_append, embeddings=queries_embeddings_train, use_only_query_embeddings=False)
        correct_matches_percentage_with_embeddings = calculate_percentage(correct_matches_with_embeddings, len(queries))
        if correct_matches_percentage_with_embeddings > base_score:
            print(correct_matches_percentage_with_embeddings, base_score, indices_append, append_coeff, len(indices_append))
            base_score = correct_matches_percentage_with_embeddings
            best_index = current_index
        else:
            candidate_indices.remove(i)
    
    if best_index is not None:
        greedy_embeddings.append(best_index)
        #max out lr for that dimention.
        embeddings_dict = {item["corpus_id"]: np.array(item["embedding"]) for item in corpus_embeddings}
        while True:
            append_coeff[best_index] += 1
            #extract the indices worth of embeddings and overwrite embeddings_dict.
            for key in embeddings_dict:
                embeddings_dict[key] = [embeddings_dict[key][index]*append_coeff[i]*lr for i, index in enumerate(greedy_embeddings)]
            sparse_embeddings = [sparse.csr_matrix(embeddings_dict[item["corpus_id"]]) for item in corpus_data]
            embeddings_sparse_matrix = sparse.vstack(sparse_embeddings)
            X = sparse.hstack([corpus_vectors, embeddings_sparse_matrix], format="csr")
            # X = normalize_embedding_0(X, indices_append, axis=1)
            #Todo: send the LR array to the score function too to apply to corpus. 
            #ToDo
            correct_matches_with_embeddings, query_results_with_embeddings, _, _ = process_queries(queries, queries_data_train, X, tsv_file, vectorizer, lr=lr, append_coeff=append_coeff,indices=indices_append, embeddings=queries_embeddings_train, use_only_query_embeddings=False)
            correct_matches_percentage_with_embeddings = calculate_percentage(correct_matches_with_embeddings, len(queries))
            if correct_matches_percentage_with_embeddings > base_score:
                print(correct_matches_percentage_with_embeddings, base_score, indices_append, append_coeff, len(indices_append))
                base_score = correct_matches_percentage_with_embeddings
            else:
                append_coeff[best_index] -= 1
                break

    best_index = None
print(greedy_embeddings)
percentage_results["With Greedy Embeddings"] = base_score

"""
print("With BM25:")
X = corpus_vectors
_, _, bm25_correct_matches, bm25_query_results = process_queries(queries, queries_data, X, tsv_file, vectorizer, bm25=bm25, use_only_query_embeddings=False)
bm25_correct_matches_percentage = calculate_percentage(bm25_correct_matches, len(queries))
print(f"Percentage of correct matches: {bm25_correct_matches_percentage}%\n")
percentage_results["With BM25"] = bm25_correct_matches_percentage
"""

"""
print("With Only Embeddings:")
X = normalize(embeddings_sparse_matrix, axis=1)
correct_matches_only_embeddings, query_results_only_embeddings, _, _ = process_queries(queries, queries_data, X, tsv_file, vectorizer, embeddings=queries_embeddings_train, use_only_query_embeddings=True)
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
"""