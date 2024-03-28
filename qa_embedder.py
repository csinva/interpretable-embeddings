import sglang as sgl
import numpy as np
from typing import List
import json
import argparse
import os
from os.path import join, expanduser
from tqdm import tqdm
import guidance
import torch
import imodelsx.llm
import qa_questions


class QuestionEmbedder:
    def __init__(
            self,
            questions: List[str] = qa_questions.get_questions(),
            # checkpoint: str = 'mistralai/Mixtral-8x7B-v0.1',
            checkpoint: str = 'mistralai/Mistral-7B-v0.1',
            prompt: str = 'Input: {example}\nQuestion: {question} Answer yes or no.\nAnswer:',
    ):
        self.questions = questions
        self.prompt = prompt
        # self.llm = guidance.models.Transformers("meta-llama/Llama-2-13b-hf")
        self.llm = imodelsx.llm.get_llm(
            checkpoint, CACHE_DIR=expanduser("~/cache_qa_embedder"))

    def __call__(self, examples: List[str], verbose=True) -> np.ndarray:
        embeddings = np.zeros((len(examples), len(self.questions)))
        for ex_num in tqdm(range(len(examples))):
            programs = [
                self.prompt.format(example=examples[ex_num], question=question)
                for question in self.questions
            ]
            # answers = self.llm(programs, max_new_tokens=3, verbose=verbose)
            # answers = list(map(lambda x: 'yes' in x.lower(), answers))
            answers = self.llm(programs, target_token_strs=[
                               ' yes', ' no'], return_top_target_token_str=True)
            answers = list(map(lambda x: ' yes' == x, answers))

            for i, answer in enumerate(answers):
                if answer:
                    embeddings[ex_num, i] = 1
        return embeddings


if __name__ == "__main__":
    questions = [
        'Is the input related to food preparation?',
        'Does the input mention laughter?',
    ]
    examples = ['I sliced some cucumbers', 'The kids were laughing']
    checkpoint = 'gpt2'
    # checkpoint = "meta-llama/Llama-2-7b-hf"
    # checkpoint = "meta-llama/Llama-2-7b-hf"
    # checkpoint = "mistralai/Mixtral-8x7B-v0.1"
    # checkpoint = 'mistralai/Mistral-7B-v0.1'

    # test
    llm = imodelsx.llm.get_llm(checkpoint)

    # outputs = [llm(x, use_cache=0) for x in examples]
    # # outputs2 = [llm(x, use_cache=0) for x in examples]
    # outputs2 = llm(examples)
    # for i in range(len(examples)):
    #     print('example', examples[i])
    #     print('o1', repr(outputs[i]))
    #     print('o2', repr(outputs2[i]))
    # assert outputs[i] == outputs2[i]

    # questions = qa_questions.get_questions()[:5]
    # embedder = QuestionEmbedder(questions=questions, checkpoint=checkpoint)
    embedder = QuestionEmbedder(
        questions=qa_questions.get_questions(), checkpoint=checkpoint)
    embeddings = embedder(examples)
    print(embeddings)
