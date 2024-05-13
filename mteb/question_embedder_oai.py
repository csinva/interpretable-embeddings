import sglang as sgl
import numpy as np
from typing import List
import json
import argparse
import os
from tqdm import tqdm
from openai import AzureOpenAI
from azure.identity import AzureCliCredential
from azure.keyvault.secrets import SecretClient


@sgl.function
def question_sglang(s, question: str, data: str, prompt: str):
    s += sgl.system("You are a helpful assistant who gives unbiased answers. Please only answer in yes or no.")
    s += sgl.user("\n Query: " + data + "\n Question: " + question + "\n Answer: ")
    s += sgl.assistant(
        sgl.gen(
            "answer",
            max_tokens=20,
            stop=".",
            temperature=0,
            top_p=0.1,
        )
    )
    # s += sgl.assistant(sgl.gen("answer", max_tokens=100, stop=".", temperature=0, top_p=0.1))
    return s


# Function to generate a question using sglang
# @sgl.function
# def question_sglang(s, question: str, data: str, prompt: str):
#     s += prompt + "\n Query: " + data + "\n Question: " + question + "\n Answer: "
#     s += sgl.gen(
#         "answer", max_tokens=20, stop=".", temperature=0, top_p=0.1, regex="(yes)|(no)"
#     )
#     # s += sgl.gen("answer", max_tokens=100, stop=".", temperature=0, top_p=0.1)
#     return s


# Class to embed questions
class QuestionEmbedder:
    def __init__(self, questions: List[str], prompt: str):
        self.questions = questions
        self.prompt = prompt

    def __call__(self, data: List[str]) -> np.ndarray:
        outputs = question_sglang.run_batch(
            [
                {"question": q, "data": d, "prompt": self.prompt}
                for d in data
                for q in self.questions
            ],
            progress_bar=True
        )
        assert len(outputs) == len(self.questions)
        return outputs


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--key",
        choices=["query_id", "corpus_id"],
        default="query_id",
        help="key to determine if embedding queries or corpus",
    )
    parser.add_argument(
        "--tq",
        type=int,
        default=None,
        help="limit for the number of queries or corpus data to process",
    )
    parser.add_argument(
        "--nq",
        type=int,
        default=None,
        help="limit for the number of questions to process",
    )
    parser.add_argument(
        "--query_questions",
        default="questions/msmarco/questions_query.json",
        help="path to the query questions file",
    )
    parser.add_argument(
        "--query_file",
        default="questions/msmarco/sample_dev_queries_100_beir.json",
        help="path to the query file",
    )
    parser.add_argument(
        "--corpus_questions",
        default="questions/msmarco/questions_corpus.json",
        help="path to the corpus questions file",
    )
    parser.add_argument(
        "--corpus_file",
        default="questions/msmarco/sample_dev_corpus_120_beir.json",
        help="path to the corpus file",
    )
    parser.add_argument("--out_file", default="dev_out", help="path to the out file")
    return parser.parse_args()


def load_data(file_path, limit=None):
    with open(file_path, "r") as file:
        data = json.load(file)
        if limit is not None:
            data = data[:limit]
    return data


def save_data(data, out_file):
    with open(out_file, "w") as f:
        json.dump(data, f, indent=4)


def prepare_output_files(existing_data, out_file, key, questions):
    embeddings_out_file = "embeddings_" + out_file
    positive_questions_out_file = "positive_questions_" + out_file

    # Remove the files if they already exist
    if os.path.exists(embeddings_out_file):
        os.remove(embeddings_out_file)
    if os.path.exists(positive_questions_out_file):
        os.remove(positive_questions_out_file)

    # Prepare data for output files
    embeddings_data = [
        {key: data[key], "embedding": [e["answer"] for e in data["embedding"]]}
        for data in existing_data
    ]
    positive_questions_data = [
        {
            key: data[key],
            "query": data["query"],
            "embedding": [
                {
                    "question_id": e["question_id"],
                    "question": questions[e["question_id"] - 1]["question"],
                    "answer": e["answer"],
                }
                for e in data["embedding"]
                if e["answer"] == 1
            ],
        }
        for data in existing_data
    ]

    # Save output files
    with open(embeddings_out_file, "w") as f:
        json.dump(embeddings_data, f, indent=4)

    with open(positive_questions_out_file, "w") as f:
        json.dump(positive_questions_data, f, indent=4)


def process_data(data, embedder, key, existing_data, out_file):
    existing_ids = [data[key] for data in existing_data]
    data_type = "query" if key == "query_id" else "corpus"

    for d in tqdm(data, desc="Processing data"):
        if d[key] not in existing_ids:
            output = embedder([d["query"] if key == "query_id" else d["corpus"]])
            answer = [output[i]["answer"] for i in range(len(output))]
            invalid_values = [a for i, a in enumerate(answer) if a.lower() not in ["yes", "no"]]
            if invalid_values:
                raise ValueError(
                    f"Invalid answer values: {invalid_values} at indices {[i for i, a in enumerate(answer) if a in invalid_values]}"
                )
            embedding = [1 if a.lower() == "yes" else 0 for a in answer]
            # print(embedding)

            # Compute the values to append to the JSON separately
            id = d[key]
            text = d["query"] if key == "query_id" else d["corpus"]
            embedding_dict = [
                {"question_id": i + 1, "answer": embedding[i]}
                for i in range(len(embedding))
            ]

            # Append the computed values to the JSON file
            existing_data.append(
                {key: id, data_type: text, "embedding": embedding_dict}
            )
            save_data(existing_data, out_file)
    return existing_data


# Main function
if __name__ == "__main__":
    # sgl.set_default_backend(sgl.RuntimeEndpoint("http://localhost:30000"))

    credential = AzureCliCredential()
    secret_client = SecretClient(
        "https://eywa-secrets.vault.azure.net", credential=credential
    )

    backend = sgl.OpenAI(
        model_name="gpt-4",
        api_version="2024-02-15-preview",
        azure_endpoint=secret_client.get_secret("ks-demo-proxy-endpoint").value,
        api_key=secret_client.get_secret("agcopilot-proxy-key").value,
        is_azure=True,
    )

    sgl.set_default_backend(backend)

    args = parse_arguments()

    # File paths
    file_paths = {
        "query_id": (args.query_questions, args.query_file),
        "corpus_id": (args.corpus_questions, args.corpus_file),
    }

    key = args.key
    out_file = (
        args.out_file + ("_queries" if key == "query_id" else "_corpus") + ".json"
    )

    # Check for valid key
    if args.key not in file_paths:
        raise ValueError(
            "Invalid key. Please provide either 'query_id' or 'corpus_id'."
        )

    # Load questions and data
    questions = load_data(file_paths[key][0], args.nq)
    data = load_data(file_paths[key][1], args.tq)

    # Initialize embedder
    prompt = ""
    embedder = QuestionEmbedder([q["question"] for q in questions], prompt)

    # Load existing data if any
    existing_data = []
    if os.path.exists(out_file):
        existing_data = load_data(out_file)

    # Process data
    existing_data = process_data(data, embedder, key, existing_data, out_file)

    # Prepare output files
    prepare_output_files(existing_data, out_file, key, questions)
