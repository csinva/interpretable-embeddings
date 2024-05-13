import logging
import json
import os
from mteb import MTEB
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sentence_transformers import SentenceTransformer
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
from openai import AzureOpenAI
from azure.identity import AzureCliCredential
from azure.keyvault.secrets import SecretClient
import pandas as pd
from random import sample

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("main")

def set_openai_parameters():
    credential = AzureCliCredential()
    secret_client = SecretClient(
        "https://eywa-secrets.vault.azure.net", credential=credential
    )
    endpoint = secret_client.get_secret("ks-demo-proxy-endpoint").value
    api_key = secret_client.get_secret("agcopilot-proxy-key").value
    client = AzureOpenAI(
        api_version="2024-02-15-preview",
        api_key=api_key,
        azure_endpoint=endpoint,
    )
    return client

class QAembedder(SentenceTransformer):
    """Class that uses OpenAI's models to generate embeddings for question-answering tasks."""

    def __init__(
        self,
        questions_file,
        model_name="gpt-4",
        cache_file="QAembedderCache.json",
        system_prompt_file="QAembedder_system_prompt.txt",
        batch_size=100,
    ):
        super().__init__(model_name_or_path=None)
        self.client = set_openai_parameters()
        self.model_name = model_name
        self.cache_file = cache_file
        self.questions_file = questions_file
        self.system_prompt_file = system_prompt_file
        self.batch_size = batch_size
        if os.path.exists(cache_file):
            with open(cache_file, "r") as f:
                self.cache = json.load(f)
        else:
            self.cache = {}

    def fit(self, texts):
        # nothing happens here
        return self

    def encode(self, texts, **kwargs):
        responses_all = []
        with open(self.questions_file, "r") as f:
            questions = json.load(f)
        with open(self.system_prompt_file, "r") as f:
            system_prompt = f.read()
        for text in texts:
            if text in self.cache:
                responses_all.append(np.array(self.cache[text]))
                continue
            system_prompt = system_prompt + "\n" + text
            responses = []
            for i in range(0, len(questions), self.batch_size):
                batch_questions = questions[i : i + self.batch_size]
                user_prompt = [question["question"] for question in batch_questions]
                user_prompt = "Questions:\n" + str(user_prompt)
                message = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ]
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=message,
                    max_tokens=1000,
                    temperature=0,
                    top_p=0,
                    timeout=600,
                )
                # Assumes model output looks like a list. Convert the string that looks like a list into an actual list
                # TODO: use sglang, or guidance for structured generation.
                model_output_str = response["choices"][0]["message"]["content"]
                model_output = [
                    int(i) for i in model_output_str.strip("][").split(", ")
                ]

                if not all(val in [0, 1] for val in model_output):
                    print(f"Unexpected model output: {model_output}")
                    raise ValueError("Model output should only contain 0 or 1.")
                responses.extend(model_output)
            self.cache[text] = responses
            responses_all.append(np.array(responses))
        with open(self.cache_file, "w") as f:
            json.dump(self.cache, f)
        return np.array(responses_all)

    def transform(self, texts):
        return self.encode(texts)


def extract_queries_from_dataset(dataset_path, split="test", num_queries=10):
    qrels_path = os.path.join(dataset_path, "qrels", f"{split}.tsv")
    qrels_df = pd.read_csv(qrels_path, sep='\t', header=None, names=["query_id", "corpus_id", "score"], skiprows=1)
    unique_queries = list(set(qrels_df["query_id"].values))
    
    if len(unique_queries) < num_queries:
        num_queries = len(unique_queries)
        
    sampled_queries = sample(unique_queries, num_queries)
    sampled_queries = [int(query) for query in sampled_queries]  # Convert sampled_queries to int

    queries_path = os.path.join(dataset_path, "queries.jsonl")
    queries_df = pd.read_json(queries_path, lines=True)
    queries_df.rename(columns={"_id": "query_id", "text": "query"}, inplace=True)
    queries_df['query_id'] = queries_df['query_id'].astype(int)  # Convert query_id to int to match datatype of sampled_queries

    extracted_queries = queries_df[queries_df["query_id"].isin(sampled_queries)].to_dict('records')
    return extracted_queries, num_queries

vectorizers = {
    "QAembedder": QAembedder("../questions/msmarco/output_beir_400_train.json"),
}

if __name__ == "__main__":
    for model_name, model in vectorizers.items():
        logger.info(f"Encoding queries with model: {model_name}")
        queries, _ = extract_queries_from_dataset("../questions/msmarco/msmarco-dataset", "test")
        queries_text = [query["query"] for query in queries]
        model.encode(queries_text)
