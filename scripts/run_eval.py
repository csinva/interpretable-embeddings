import argparse
import json
import requests


from d3 import TASKS_D3


def main(args):
    # TODO check cache/end

    df = TASKS_D3[args.model]['gen_func'](return_df=True)

    # TODO save to cache


def parse_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'task',
        type=str,
        choices=TASKS_D3.keys()
    )
    parser.add_argument(
        '--model',
        default='mistral-7b',
        choices=['mistral-7b', 'llama3-8b', 'mixtral-8x22b', 'gpt4'],
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args)