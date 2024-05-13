import json
from datasets import load_dataset

def extract_queries_from_dataset(split="test", num_queries=100, params=["query_id", "query"]):
    """
    Extracts queries from the msmarco dataset.

    Args:
        split (str): The split of the dataset to extract queries from. Default is 'test'.
        num_queries (int): The number of queries to extract. Default is 100.
        params (list): The parameters to extract for each query. Default is ['query_id', 'query'].

    Returns:
        list: A list of dictionaries, each containing the specified parameters for a query.
    """
    dataset = load_dataset("ms_marco", "v2.1")
    extracted_queries = [
        {param: dataset[split][i][param] for param in params}
        for i in range(num_queries)
    ]
    return extracted_queries

def save_data_to_json_file(data, filename="sample_queries_100_hf.json"):
    """
    Writes data to a JSON file.

    Args:
        data (list): The data to write to the file.
        filename (str): The name of the file to write to.
    """
    with open(filename, "w") as file:
        json.dump(data, file, indent=4)

if __name__ == "__main__":
    queries = extract_queries_from_dataset()
    save_data_to_json_file(queries)
