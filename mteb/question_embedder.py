import sglang as sgl
import numpy as np
from typing import List
import json
import argparse
import os
from tqdm import tqdm
import time


# Function to generate a question using sglang
@sgl.function
def question_sglang(s, question: str, data: str, prompt: str, key: str):
    data_type = "\n Query: " if key=="query_id" else "\n Corpus: "
    # s += prompt + data_type + data + "\n Question: " + question + "\n Answer: "
    s += prompt + data_type + data + "\n Question: " + question + "\n Objective Reasoning: "
    s += sgl.gen("reason", max_tokens=120, stop=".", temperature=0, top_p=0.1)
    s += "Answer: "
    s += sgl.gen("answer", max_tokens=10, stop=".", temperature=0, top_p=0.1, choices=['yes', 'no'])
    return s


def embedder(data: List[str], questions: List[str], prompt: str, key: str) -> np.ndarray:
    outputs = question_sglang.run_batch(
        [
            {"question": q, "data": d, "prompt": prompt, "key": key}
            for d in data
            for q in questions
        ],
        num_threads=2048,
        # progress_bar=True
    )
    assert len(outputs) == len(questions)
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
        "--nq",
        type=int,
        default=None,
        help="limit for the number of questions to process",
    )
    parser.add_argument(
        "--query_questions",
        default="questions/msmarco/questions_query_2k.json",
        help="path to the query questions file",
    )
    parser.add_argument(
        "--query_file",
        default="questions/msmarco/sample_dev_queries_all_beir.json",
        help="path to the query file",
    )
    parser.add_argument(
        "--corpus_questions",
        default="questions/msmarco/questions_corpus_2k.json",
        help="path to the corpus questions file",
    )
    parser.add_argument(
        "--corpus_file",
        default="questions/msmarco/sample_dev_corpus_all_beir.json",
        help="path to the corpus file",
    )
    parser.add_argument("--out_file", default="dev_out", help="path to the out file")
    parser.add_argument("-s", "--start_index", required=True, type=int, help="Start index for processing data")
    parser.add_argument("-e", "--end_index", required=True ,type=int, help="End index for processing data")
    return parser.parse_args()


def load_data(file_path, start_index=None, end_index=None):
    with open(file_path, "r") as file:
        data = json.load(file)
        if start_index is not None and end_index is not None:
            data = data[start_index:end_index]
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

    data_type = "query" if key=="query_id" else "corpus"
    positive_questions_data = [
        {
            key: data[key],
            data_type: data[data_type],
            "embedding": [
                {
                    "question_id": e["question_id"],
                    "question": questions[e["question_id"] - 1]["question"],
                    "answer": e["answer"],
                    "reason": str(e["reason"]),
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


def process_data(data, prompt, key, out_file, questions):
    data_type = "query" if key == "query_id" else "corpus"
    processed_data = []
    print("Starting data processing...")
    for index, d in enumerate(tqdm(data, desc="Processing data")):
        start_time = time.time()
        output = embedder([d["query"] if key == "query_id" else d["corpus"]], [q["question"] for q in questions], prompt, key)
        print(time.time() - start_time)
        answer = [output[i]["answer"] for i in range(len(output))]
        reason = [output[i]["reason"] for i in range(len(output))]
        
        invalid_values = [a for i, a in enumerate(answer) if a not in ["yes", "no"]]
        if invalid_values:
            raise ValueError(
                f"Invalid answer values: {invalid_values} at indices {[i for i, a in enumerate(answer) if a in invalid_values]}"
            )
        embedding = [1 if a == "yes" else 0 for a in answer]

        id = d[key]
        text = d["query"] if key == "query_id" else d["corpus"]
        embedding_dict = [
            {"question_id": i + 1, "question": questions[i]["question"], "answer": embedding[i], "reason": str(reason[i])}
            for i in range(len(embedding))
        ]

        processed_data.append(
            {key: id, data_type: text, "embedding": embedding_dict}
        )
    save_data(processed_data, out_file)
    return processed_data


# Main function
if __name__ == "__main__":
    sgl.set_default_backend(sgl.RuntimeEndpoint("http://localhost:30000"))
    # sgl.set_default_backend(sgl.RuntimeEndpoint("http://localhost:10001"))

    args = parse_arguments()

    # File paths
    file_paths = {
        "query_id": (args.query_questions, args.query_file),
        "corpus_id": (args.corpus_questions, args.corpus_file),
    }

    start_index = args.start_index
    end_index = args.end_index
    print(start_index)
    print(end_index)
    key = args.key
    out_file = f"{args.out_file}_{ 'queries' if key == 'query_id' else 'corpus' }_{start_index}_{end_index}.json"

    # Check for valid key
    if args.key not in file_paths:
        raise ValueError(
            "Invalid key. Please provide either 'query_id' or 'corpus_id'."
        )

    # Load questions and data
    questions = load_data(file_paths[key][0], 0, args.nq)
    data = load_data(file_paths[key][1], start_index, end_index)

    # Initialize embedder
    data_type = "query" if key == "query_id" else "corpus"
    prompt = f"Analyze the {data_type} provided and respond to the question with a simple 'yes' or 'no' answer. Do not make assumptions beyond the information presented in the {data_type}. Your response must be precise and strictly answerable on the given {data_type}. Provide a clear rationale for your answer."
    # prompt = "You are a helpful assistant. Your job is to assess the" + data_type + "and answer the question correclty in yes or no. Be precise and strict."
    # prompt = "You are an helpful and unbiased assistant. You dont assume anything and your job is to answer this question for the given" + data_type + "in yes or no only along with the reasoning. First think it thru and then answer."

    # Load existing data if any
    if not os.path.exists(out_file):
        # Process data
        processed_data = process_data(data, prompt, key, out_file, questions)

        # Prepare output files
        prepare_output_files(processed_data, out_file, key, questions)

    # time for i in {1..1000}; do timeout 6m python question_embedder.py --key "corpus_id"; done
    # time for i in {1..1400}; do timeout 3m python question_embedder.py; done