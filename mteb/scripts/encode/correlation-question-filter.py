"""
File: correlation-question-filter.py
Description: Filters questions based on the correlation of query embeddings to reduce redundancy and improve the quality of question sets.
Usage: Run this script with the required arguments to filter questions based on embeddings' correlation.
"""

import json
import argparse

import numpy as np


if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Filter questions based on embeddings' correlation.")
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', default="runs/mixtral-dev-all/embeddings_dev_out_corpus_all.json", type=str, help='Path to query embeddings file')
    parser.add_argument('--q_file', default="../../questions/msmarco/questions_query_340.json", type=str, help='Path to query questions file')
    parser.add_argument('--c_file', default="../../questions/msmarco/questions_corpus_340.json", type=str, help='Path to corpus questions file')
    parser.add_argument('--q_em_file', default="runs/mixtral-dev-all/embeddings_dev_out_queries_all.json", type=str, help='Path to query embedding file')
    parser.add_argument('--c_em_file', default="runs/mixtral-dev-all/embeddings_dev_out_corpus_all.json", type=str, help='Path to corpus embedding file')
    parser.add_argument('--q_out_file', default="../../questions/msmarco/filtered_questions_query.json", type=str, help='Path to filtered query questions file')
    parser.add_argument('--c_out_file', default="../../questions/msmarco/filtered_questions_corpus.json", type=str, help='Path to filtered corpus questions file')
    parser.add_argument('--q_em_out_file', default="./filtered_embeddings_dev_out_queries.json", type=str, help='Path to filtered query embedding file')
    parser.add_argument('--c_em_out_file', default="./filtered_embeddings_dev_out_corpus.json", type=str, help='Path to filtered corpus embedding file')

    args = parser.parse_args()

    with open(args.file, 'r') as file:
        data = json.load(file)
    with open(args.q_file, 'r') as file:
        q_questions = json.load(file)
    with open(args.c_file, 'r') as file:
        c_questions = json.load(file)

    embeddings = [data[i]["embedding"] for i in range(len(data))]

    embeddings = np.array(embeddings)
    n_features_embeddings = embeddings.shape[1]
    std_devs = np.std(embeddings, axis=0)
    std_indices_to_keep = np.where(std_devs != 0)[0]
    std_indices_to_remove = np.where(std_devs == 0)[0]
    filtered_embeddings = embeddings[:, std_indices_to_keep]
    
    # Calculate the correlation matrix for the filtered embeddings. 
    # Setting rowvar=False treats each column as a variable and each row as an observation.
    # Calculate the correlation matrix for the filtered embeddings.
    # The correlation values range from -1 to 1, where 1 means perfect positive correlation,
    # -1 means perfect negative correlation, and 0 means no correlation.
    correlation_matrix = np.corrcoef(filtered_embeddings, rowvar=False)

    threshold = 0.99
    n_features = correlation_matrix.shape[0]
    highly_correlated_pairs = []

    for i in range(n_features):
        for j in range(i+1, n_features):  # Avoid self-correlation and duplicates
            if abs(correlation_matrix[i, j]) > threshold:
                highly_correlated_pairs.append((i, j))
    
    # print(highly_correlated_pairs)
    # for i,j in highly_correlated_pairs:
    #     print(q_questions[std_indices_to_keep[i]])
    #     print(q_questions[std_indices_to_keep[j]])
    #     print("------")
    # for i in std_indices_to_remove:
    #     print(q_questions[i])

    additional_indices_to_remove = sorted(list(set([std_indices_to_keep[i] for i, j in highly_correlated_pairs])))
    indices_to_remove = sorted(np.concatenate((std_indices_to_remove, additional_indices_to_remove)).astype(np.int64))
    indices_to_keep = set(range(n_features_embeddings))-{i for i in indices_to_remove}

    print("old #questions = ", n_features_embeddings)
    print("new #questions = ", len(indices_to_keep))

    filtered_q_questions = [{"question_id": q_questions[index]["question_id"], "question": q_questions[index]["question"]} for i, index in enumerate(indices_to_keep)]
    filtered_c_questions = [{"question_id": c_questions[index]["question_id"], "question": c_questions[index]["question"]} for i, index in enumerate(indices_to_keep)]

    with open(args.q_out_file, "w") as file:
        json.dump(filtered_q_questions, file, indent=4)
        print("output written to: ", args.q_out_file)
    
    with open(args.c_out_file, "w") as file:
        json.dump(filtered_c_questions, file, indent=4)
        print("output written to: ", args.c_out_file)

    """
    Filter Embeddings
    """
    # Load the embeddings from q_em_file and c_em_file
    with open(args.q_em_file, 'r') as file:
        q_em_data = json.load(file)
    with open(args.c_em_file, 'r') as file:
        c_em_data = json.load(file)

    # Remove the indices_to_remove from the embeddings list in each dict
    for dict in q_em_data:
        for index in sorted(indices_to_remove, reverse=True):
            del dict["embedding"][index]
        assert len(dict["embedding"]) == len(indices_to_keep)
    for dict in c_em_data:
        for index in sorted(indices_to_remove, reverse=True):
            del dict["embedding"][index]
        assert len(dict["embedding"]) == len(indices_to_keep)

    # Write the updated embeddings back to q_em_file and c_em_file
    with open(args.q_em_out_file, "w") as file:
        json.dump(q_em_data, file, indent=4)
        print("Filtered query embeddings written to: ", args.q_em_out_file)

    with open(args.c_em_out_file, "w") as file:
        json.dump(c_em_data, file, indent=4)
        print("Filtered corpus embeddings written to: ", args.c_em_out_file)
