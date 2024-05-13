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
import sglang as sgl


def read_json_file(filename):
    with open(filename, "r") as f:
        data = json.load(f)
    return data


def read_prompt_file(filename):
    with open(filename, "r") as f:
        prompt = f.read()
    return prompt

def set_parameters(mode):
    if mode == "openai":
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
    elif mode == "local":
        sgl.set_default_backend(sgl.RuntimeEndpoint("http://localhost:30000"))
    else:
        raise ValueError("mode = " + mode + " not supported")


def call_model(batch_queries, system_prompt, model, mode, client):
    if mode=="openai":
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
                # response_format={ "type": "json_object" },
            )
            res = response.choices[0].message.content
            return res
        except Exception as E:
            print(E)
            return None
    elif mode=="local":
        @sgl.function
        def qa(s, query):
            s+= system_prompt + "\n" + "Input Json: \n" + query + "\n" + "Output Json: \n "+ sgl.gen("output", max_tokens=4000, temperature=0, top_p=0.1, stop="\n\n")
        processed_batch_queries = [{"query": str(query)} for query in batch_queries]
        response = qa.run_batch(processed_batch_queries)
        processed_response = [eval(res["output"]) for i, res in enumerate(response)]
        return str(processed_response)


def append_to_json(data, filename):
    try:
        existing_data = read_json_file(filename)
    except FileNotFoundError:
        existing_data = []
    existing_data.extend(data)
    with open(filename, "w") as f:
        json.dump(existing_data, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", default="sample_queries_400_beir_train.json")
    parser.add_argument("--prompt_file", default="system_prompt.txt")
    parser.add_argument("--tq", default=None, type=int, help="Total queries to process from the input file. If not provided, all queries will be processed.")
    parser.add_argument("--output_file", default="output_beir_400_train.json")
    parser.add_argument("--model", default="gpt-4-32k")
    parser.add_argument("--sort_key", default="query_id")
    parser.add_argument("--batch_size", default=20, type=int)
    parser.add_argument("--mode", default="openai")
    args = parser.parse_args()

    if os.path.exists(args.output_file):
        os.remove(args.output_file)

    queries = read_json_file(args.input_file)
    if args.tq is not None:
        queries = queries[:args.tq]
    system_prompt = read_prompt_file(args.prompt_file)
    pbar = tqdm(total=len(queries), desc="Processing queries")
    api_calls = 0
    client = set_parameters(mode=args.mode)
    max_workers = 1 if args.mode=="local" else 5
    batch_size = 4 if args.mode=="local" else args.batch_size

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for i in range(0, len(queries), batch_size):
            batch_queries = queries[i:i+batch_size]
            futures.append(executor.submit(call_model, batch_queries, system_prompt, args.model, args.mode, client))

        for future in as_completed(futures):
            try:
                response = future.result()
                api_calls += 1
                if response is not None:
                    append_to_json(ast.literal_eval(response), args.output_file)
                else:
                    print("Response is None")
            except Exception as exc:
                print(f'Batch query generated an exception: {exc}')
            finally:
                pbar.update(batch_size)

    pbar.close()
    print(f'Total API calls made: {api_calls}')

    # Post-processing to ensure the order of queries in the output file is the same as the input file
    input_queries = read_json_file(args.input_file)
    output_data = read_json_file(args.output_file)
    output_data.sort(key=lambda x: input_queries.index(next(item for item in input_queries if item[args.sort_key] == x[args.sort_key])))
    with open(args.output_file, "w") as f:
        json.dump(output_data, f, indent=4)