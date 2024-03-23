import sglang as sgl
import numpy as np
from typing import List
import json
import argparse
import os
from tqdm import tqdm
import guidance
import torch
import imodelsx.llm
import qa_questions


class QuestionEmbedder:
    def __init__(
            self,
            questions: List[str],
            checkpoint: str = 'mistralai/Mixtral-8x7B-v0.1',
            prompt: str = 'Input: {example}\nQuestion: {question} Answer yes or no.\nAnswer:',
    ):
        self.questions = questions
        self.prompt = prompt
        # self.llm = guidance.models.Transformers("meta-llama/Llama-2-13b-hf")
        self.llm = imodelsx.llm.get_llm(checkpoint)

    def __call__(self, examples: List[str]) -> np.ndarray:
        embeddings = np.zeros((len(examples), len(self.questions)))
        for ex_num, example in enumerate(examples):
            programs = [
                self.prompt.format(example=example, question=question)
                for question in self.questions
            ]
            # program = self.prompt.format(
            # example=example, question=question)
            # program = self.prompt.format(
            # example=example, question=question) + guidance.select([" yes", " no"], name='ans')
            # ans = (self.llm + program)['ans']

            answers = self.llm(programs, max_new_tokens=3)
            answers = list(map(lambda x: 'yes' in x.lower(), answers))

            for i, answer in enumerate(answers):
                if answer:
                    embeddings[ex_num, i] = 1
            # print('\n\nPROMPT', program)
            # print('ANS', ans)
            # if ans == ' yes':
            # if 'yes' in ans.lower():

            # embeddings[examples.index(
                # example), questions.index(question)] = 1
        return embeddings


if __name__ == "__main__":
    questions = [
        'Is the input related to food preparation?',
        'Does the input mention laughter?',
    ]
    examples = ['Roses are red, violets are',
                'I sliced some cucumbers', 'The kids were laughing']
    # checkpoint = 'gpt2'
    # checkpoint = "meta-llama/Llama-2-7b-hf"
    checkpoint = "meta-llama/Llama-2-7b-hf"
    # checkpoint = "mistralai/Mixtral-8x7B-v0.1"

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
    embedder = QuestionEmbedder(questions=questions, checkpoint=checkpoint)
    embeddings = embedder(examples)
    print(embeddings)
