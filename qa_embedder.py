import sglang as sgl
import numpy as np
from typing import List
import json
import argparse
import os
from tqdm import tqdm
import guidance
import imodelsx.llm
import qa_questions


class QuestionEmbedder:
    def __init__(
            self, questions: List[str],
            prompt: str = 'Input: {example}\nQuestion: {question} Answer yes or no.\nAnswer:',
            checkpoint: str = 'mistralai/Mixtral-8x7B-v0.1',
    ):
        self.questions = questions
        self.prompt = prompt
        # self.llm = guidance.models.Transformers("meta-llama/Llama-2-13b-hf")
        self.llm = imodelsx.llm.get_llm(checkpoint)

    def __call__(self, examples: List[str]) -> np.ndarray:
        embeddings = np.zeros((len(examples), len(self.questions)))
        for example in examples:
            for question in questions:
                program = self.prompt.format(
                    example=example, question=question)
                # program = self.prompt.format(
                # example=example, question=question) + guidance.select([" yes", " no"], name='ans')
                # ans = (self.llm + program)['ans']

                ans = self.llm(program)
                print('\n\nPROMPT', program)
                print('ANS', ans)
                # if ans == ' yes':
                if 'yes' in ans.lower():
                    embeddings[examples.index(
                        example), questions.index(question)] = 1
        return embeddings


if __name__ == "__main__":
    # questions = [
    # 'Is the input related to food preparation?',
    # 'Does the input mention laughter?',
    # ]
    questions = qa_questions.get_questions()
    examples = ['I sliced some cucumbers', 'The kids were laughing']
    # checkpoint = "meta-llama/Llama-2-7b-hf"
    checkpoint = "mistralai/Mixtral-8x7B-v0.1"
    embedder = QuestionEmbedder(examples)
    embeddings = embedder(examples)
    print(embeddings)
