"""
This script generates responses to queries using the GPT-4 model hosted on Azure. It reads queries from a JSON file, sends them to the GPT-4 model for processing, and saves the responses in another JSON file.
"""

import json
import argparse
import os
from openai import AzureOpenAI
from azure.identity import AzureCliCredential
from azure.keyvault.secrets import SecretClient
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import ast


def read_json_file(filename):
    with open(filename, "r") as f:
        data = json.load(f)
    return data


def read_prompt_file(filename):
    with open(filename, "r") as f:
        prompt = f.read()
    return prompt


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


def call_gpt4(batch_queries, system_prompt, model):
    client = set_openai_parameters()
    message = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": str(batch_queries)},
    ]
    try:
        response = client.chat.completions.create(
            model=model,
            messages=message,
            max_tokens=4000,
            temperature=0,
            top_p=0,
            timeout=600,
            response_format={"type": "json_object"},
        )
        res = response.choices[0].message.content
        return res
    except Exception as E:
        print(E)
        return None


def append_to_json(data, filename, key):
    try:
        existing_data = read_json_file(filename)
    except FileNotFoundError:
        existing_data = []

    for new_data in data:
        for existing_query in existing_data:
            if existing_query[key] == new_data[key]:
                existing_query["embedding"].extend(new_data["embedding"])
                existing_query["embedding"] = sorted(
                    existing_query["embedding"], key=lambda x: x["question_id"]
                )
                break
        else:
            new_data["embedding"] = sorted(
                new_data["embedding"], key=lambda x: x["question_id"]
            )
            existing_data.append(new_data)

    with open(filename, "w") as f:
        json.dump(existing_data, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_file",
        default="../../questions/msmarco/sample_dev_queries_100_beir.json",
    )
    parser.add_argument(
        "--questions_file", default="../../questions/msmarco/questions_query.json"
    )
    parser.add_argument(
        "--prompt_file", default="prompts/QAembedder_system_prompt_queries.txt"
    )
    parser.add_argument(
        "--tq",
        default=None,
        type=int,
        help="Total queries to process from the input file. If not provided, all queries will be processed.",
    )
    parser.add_argument(
        "--nq",
        default=None,
        type=int,
        help="Number of questions to process from the questions file. If not provided, all questions will be processed.",
    )
    parser.add_argument("--output_file", default="test_out.json")
    parser.add_argument("--model", default="gpt-4-1106-preview")
    parser.add_argument("--key", default="query_id")
    parser.add_argument("--batch_size", default=20, type=int)
    args = parser.parse_args()

    queries = read_json_file(args.input_file)
    if args.tq is not None:
        queries = queries[: args.tq]
    system_prompt = read_prompt_file(args.prompt_file)
    pbar = tqdm(total=len(queries), desc="Processing queries")
    api_calls = 0

    key = args.key

    question_dicts = read_json_file(args.questions_file)
    if args.nq is not None:
        question_dicts = question_dicts[: args.nq]

    if os.path.exists(args.output_file):
        existing_data = read_json_file(args.output_file)
    else:
        existing_data = []

    for query in queries:
        if any(d[key] == query[key] for d in existing_data):
            continue
        
        new_system_prompt = system_prompt + "\n" + str(query)
        batched_futures = []
        with ThreadPoolExecutor(max_workers=4) as executor:
            for i in range(0, len(question_dicts), args.batch_size):
                batch_questions = question_dicts[i : i + args.batch_size]
                # print("system prompt: \n", new_system_prompt)
                batched_futures.append(
                    executor.submit(
                        call_gpt4, batch_questions, new_system_prompt, args.model
                    )
                )
            responses = []
            for future in as_completed(batched_futures):
                response = future.result()
                api_calls += 1
                if response is not None:
                    try:
                        new_response = ast.literal_eval(response)
                        responses.append(new_response)
                    except Exception as e:
                        print(f"Error in ast.literal_eval: {e}")
                        print(f"Value of response: {response}")
                else:
                    print("Warning: Response is None")
                pbar.update(args.batch_size)
            append_to_json(responses, args.output_file, key)

    pbar.close()
    print(f"Total API calls made: {api_calls}")

    output_data = read_json_file(args.output_file)
    question_data = read_json_file(args.questions_file)
    question_dict = {q["question_id"]: q["question"] for q in question_data}
    positive_questions = []
    for query in output_data:
        positive_embeddings = []
        for embedding in query["embedding"]:
            question_id = embedding["question_id"]
            embedding["question"] = question_dict[question_id]
            if embedding["answer"] == 1:
                positive_embeddings.append(embedding)
        if positive_embeddings:
            positive_query = query.copy()
            positive_query["embedding"] = positive_embeddings
            positive_questions.append(positive_query)
    positive_questions_file = "positive_questions_" + args.output_file
    if os.path.exists(positive_questions_file):
        os.remove(positive_questions_file)
    with open(positive_questions_file, "w") as f:
        json.dump(positive_questions, f, indent=4)

    embeddings = []
    for query in output_data:
        embedding = [e["answer"] for e in query["embedding"]]
        if len(embedding) != 340:
            print(
                f"Error: The number of items in the embedding for query '{query[key]}' is {len(embedding)}, not 340."
            )
        embeddings.append({key: query[key], "embedding": embedding})
    embeddings_file = "embeddings_" + args.output_file
    if os.path.exists(embeddings_file):
        os.remove(embeddings_file)
    with open(embeddings_file, "w") as f:
        json.dump(embeddings, f, indent=4)
