import argparse
import json
import os
import random
import requests
import time

from openai import OpenAI
import pandas as pd

from d3 import TASKS_D3


def ask_gpt4(s: str) -> str:
    api_key = os.getenv("OPENAI_API_KEY")  # need to fill this in
    client = OpenAI(
        api_key=api_key
    )

    response = client.chat.completions.create(  # replace this value with the deployment name you chose when you deployed the associated model.
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": s}
        ],
        temperature=0,
        max_tokens=1,
        top_p=0.95,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None
    )
    return response.choices[0].message.content


def ask_llms(s: str) -> str:
    headers = {"Authorization": "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VyX2lkIjoiODZmODIxNmMtMmE4Yi00MzI2LTk5ZDgtMGNkZGRhNDA0M2ZlIiwidHlwZSI6ImFwaV90b2tlbiJ9.IO-dlAtkECxZg7WUCVnr7erFqY7s9a2rdoL4J64WTk8"}

    url = "https://api.edenai.run/v2/text/generation"
    print(s)
    payload = {
        # "providers": "cohere",
        "providers": "openai,mistral,meta",
        "text": s,
        "temperature": 0.0,
        "max_tokens": 1,
        # "fallback_providers": ""
    }

    response = requests.post(url, json=payload, headers=headers)
    result = json.loads(response.text)
    print(result)
    time.sleep(1.0)

    return { model: answer['generated_text'] for model, answer in result.items() }


def main(args):
    random.seed(42)
    out_folder = os.path.join(
        os.path.expanduser("~/Projects"),
        "interpretable-embeddings/scripts/d3/out",
    )
    os.makedirs(out_folder, exist_ok=True)
    csv_path = os.path.join(out_folder, args.task + ".csv")
    if os.path.exists(csv_path):
        print("Found cached file; exiting.")
        exit()

    task_data = TASKS_D3[args.task]
    df = task_data['gen_func'](return_df=True)
    data = [
        'task', 'idx', 'true_label', 'pred_label', 'answer', 'model'
    ]

    print("*" * 40)
    print(args.task)
    print(df['label'].value_counts())
    
    raw_question_template = (
        'Question: ' + 
        task_data['template'] + 
        task_data['target_token'] + 
        '. Yes or No? Answer:'
    )

    few_shot_datapoints = [
        ex for _, ex in df.sample(n=args.n_few_shot).iterrows()
    ]
    few_shot_examples = [
        raw_question_template.format(input=ex['input'])
        + { 1: ' Yes', 0: ' No' }[ex['label']]
        for ex in few_shot_datapoints
    ]
    question_template = '\n'.join(few_shot_examples + [raw_question_template])

    idxs = random.choices(range(len(df)), k=args.n)
    for idx in idxs:
        ex = df.iloc[idx]
        print(idx, ex['label'])
        text_input = ex['input'].strip()
        question = question_template.format(input=text_input)
        # try:
        answers_raw = ask_llms(question)
        answers_raw['gpt4'] = ask_gpt4(question)
        for model, answer_raw in answers_raw.items():
            answer = answer_raw.strip().replace(',','').replace('.','').lower()
            true_label = ex['label']
            pred_label = { 'yes': 1, 'no': 0 }.get(answer, 0)
            data.append([args.task, idx, true_label, pred_label, answer, model])
        # except Exception as e:
        #     print("Exception =>", e)
        #     continue

    df = pd.DataFrame(data)
    df.to_csv(csv_path)

def parse_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'task',
        type=str,
        choices=TASKS_D3.keys()
    )
    parser.add_argument(
        '--n',
        type=int,
        default=20,
    )
    parser.add_argument(
        '--n_few_shot',
        type=int,
        default=5,
    )
    # parser.add_argument(
    #     '--model',
    #     default='mistral-7b',
    #     choices=['mistral-7b', 'llama3-8b', 'mixtral-8x22b', 'gpt4'],
    # )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args)