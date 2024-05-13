import json
import argparse
import pandas as pd
import os
from random import sample

def extract_queries_from_dataset(dataset_path, key, split, num_queries=None):
    """
    Extracts queries from the msmarco dataset.

    Args:
        dataset_path (str): The path to the msmarco dataset.
        split (str): The split of the dataset to extract queries from. Default is 'test'.
        num_queries (int, optional): The number of queries to extract. If None, all queries are extracted. Default is None.
        key: the key to extract.

    Returns:
        list: A list of dictionaries, each containing the query_id and query.
    """
    qrels_path = os.path.join(dataset_path, "qrels", f"{split}.tsv")
    qrels_df = pd.read_csv(qrels_path, sep='\t', header=None, names=["query_id", "corpus_id", "score"], skiprows=1)
    unique_queries = list(dict.fromkeys(qrels_df[key].values))
    
    if num_queries is not None and len(unique_queries) < num_queries:
        num_queries = len(unique_queries)
        
    top_queries = unique_queries if num_queries is None else unique_queries[:num_queries]
    top_queries = [int(query) for query in top_queries]  # Convert top_queries to int

    if key == "query_id":
        queries_path = os.path.join(dataset_path, "queries.jsonl")
        queries_df = pd.read_json(queries_path, lines=True)
        queries_df.rename(columns={"_id": key, "text": "query"}, inplace=True)
    elif key == "corpus_id":
        queries_path = os.path.join(dataset_path, "corpus.jsonl")
        queries_df = pd.read_json(queries_path, lines=True)
        queries_df.rename(columns={"_id": key, "text": "corpus"}, inplace=True)
    else:
        raise ValueError("Invalid key. Key should be either 'query_id' or 'corpus_id'")
    
    queries_df[key] = queries_df[key].astype(int)  # Convert key to int to match datatype of top_queries

    extracted_queries = queries_df[queries_df[key].isin(top_queries)].to_dict('records')
    return extracted_queries, len(top_queries)

def save_data_to_json_file(data, num_queries, split, key, filename=None):
    """
    Writes data to a JSON file.

    Args:
        data (list): The data to write to the file.
        num_queries (int): The number of queries in the data.
        split (str): The split of the dataset from which the queries were extracted.
        filename (str, optional): The name of the file to write to. If None, a default name is generated.
    """
    if filename is None:
        filename = f"sample_{split}_{key}_{num_queries if num_queries is not None else 'all'}_beir.json"

    with open(filename, "w") as file:
        if data:
            json.dump(data, file, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", default="./msmarco-dataset", help="Path to the msmarco dataset.")
    parser.add_argument("--split", default="test", choices=["test", "dev", "train"], help="The split of the dataset to extract queries from.")
    parser.add_argument("--num_queries", type=int, default=None, help="The number of queries to extract. If None, all queries are extracted.")
    parser.add_argument("--key", default="query_id", help="The key to extract.")
    parser.add_argument("--output_file", default=None, help="The name of the output file.")
    args = parser.parse_args()

    queries, num_queries = extract_queries_from_dataset(args.dataset_path, args.key, args.split, args.num_queries)
    if queries:
        save_data_to_json_file(queries, num_queries, args.split, args.key, args.output_file)

# python msmarco-beir.py --split="dev" --key="corpus_id"