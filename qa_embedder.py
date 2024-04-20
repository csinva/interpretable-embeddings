import numpy as np
from typing import List
from os.path import join, expanduser
from tqdm import tqdm
import imodelsx.llm
import qa_questions
import torch.nn
from transformers import T5ForConditionalGeneration, T5Tokenizer


class QuestionEmbedder:
    def __init__(
            self,
            questions: List[str] = qa_questions.get_questions(),
            checkpoint: str = 'mistralai/Mistral-7B-Instruct-v0.2',
            use_cache: bool = True,
    ):

        self.questions = questions
        if 'mistral' in checkpoint and 'Instruct' in checkpoint:
            self.prompt = "<s>[INST]'Input text: {example}\nQuestion: {question}\nAnswer with yes or no, then give an explanation.[/INST]"
        elif 'Meta-Llama-3' in checkpoint and 'Instruct' in checkpoint:
            self.prompt = '<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a concise, helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nInput text: {example}\nQuestion: {question}\nAnswer with yes or no, then give an explanation.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'
        else:
            self.prompt = 'Input: {example}\nQuestion: {question} Answer yes or no.\nAnswer:'
        # self.llm = guidance.models.Transformers("meta-llama/Llama-2-13b-hf")
        self.llm = imodelsx.llm.get_llm(
            checkpoint, CACHE_DIR=expanduser("~/cache_qa_embedder"))
        self.use_cache = use_cache

    def __call__(self, examples: List[str], verbose=True) -> np.ndarray:
        embeddings = np.zeros((len(examples), len(self.questions)))
        for ex_num in tqdm(range(len(examples))):
            programs = [
                self.prompt.format(example=examples[ex_num], question=question)
                for question in self.questions
            ]
            # print(programs)
            answers = self.llm(
                programs,
                max_new_tokens=2,
                verbose=verbose,
                use_cache=self.use_cache,
            )

            # for i in range(len(programs)):
            # print(i, '->', repr(answers[i][:10]))

            answers = list(map(lambda x: 'yes' in x.lower(), answers))
            # print('answers', np.sum(answers), answers)
            # note: mistral token names often start with weird underscore e.g. '▁yes'
            # so this is actually better than constrained decoding
            # answers = self.llm(programs, target_token_strs=[
            #    ' yes', ' no'], return_top_target_token_str=True)
            # answers = list(map(lambda x: ' yes' == x, answers))

            for i, answer in enumerate(answers):
                if answer:
                    embeddings[ex_num, i] = 1
        return embeddings


if __name__ == "__main__":
    questions = [
        'Is the input related to food preparation?',
        'Does the input mention laughter?',
    ]
    examples = ['I sliced some cucumbers',
                'The kids were laughing', 'walking to school I was']
    # checkpoint = 'mistralai/Mistral-7B-Instruct-v0.2'
    # checkpoint = 'mistralai/Mixtral-8x7B-Instruct-v0.1'
    # checkpoint = "meta-llama/Llama-2-7b-hf"
    # checkpoint = "meta-llama/Llama-2-70b-hf"
    # checkpoint = "mistralai/Mixtral-8x7B-v0.1"
    # checkpoint = 'mistralai/Mistral-7B-v0.1'
    checkpoint = "meta-llama/Meta-Llama-3-8B-Instruct"

    # test
    # llm = imodelsx.llm.get_llm(checkpoint)
    # prompt = 'yes or no.\nQuestion:is the sky blue?\nAnswer:'
    # probs = llm(prompt, return_next_token_prob_scores=True)
    # probs_top = probs.flatten().argsort()[::-1]
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
    # embedder = QuestionEmbedder(
    # questions=qa_questions.get_questions(), checkpoint=checkpoint)
    embedder = QuestionEmbedder(
        questions=questions, checkpoint=checkpoint, use_cache=False)
    embeddings = embedder(examples)
    print(embeddings)
