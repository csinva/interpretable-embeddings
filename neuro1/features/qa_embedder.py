import scipy.special
import numpy as np
from typing import List
from os.path import join, expanduser, dirname
from tqdm import tqdm
import imodelsx.llm
import neuro1.features.qa_questions as qa_questions
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW, AutoModel
import torch
from torch import nn
from neuro1.config import root_dir
# from vllm import LLM, SamplingParams
# import torch


class MutiTaskClassifier(nn.Module):
    def __init__(self, checkpoint, num_binary_outputs):
        super().__init__()
        self.model = AutoModel.from_pretrained(
            checkpoint, return_dict=True  # , output_hidden_states=True,
        )
        self.classifiers = nn.ModuleList(
            [nn.Linear(self.model.config.hidden_size, 2)
             for _ in range(num_binary_outputs)]
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids,
                             attention_mask=attention_mask)
        logits = torch.stack([classifier(outputs.pooler_output)
                              for classifier in self.classifiers])
        logits = logits.permute(1, 0, 2)
        return logits


class FinetunedQAEmbedder:
    def __init__(self, checkpoint, qa_questions_version='v3_boostexamples'):
        # print(f'{checkpoint=} {num_binary_outputs=}')
        self.tokenizer = AutoTokenizer.from_pretrained(
            checkpoint, return_token_type_ids=False)  # , device_map='auto')
        question_counts = {
            'v1': 376,
            'v2': 142,
            'v3_boostexamples': 156,
        }
        self.model = MutiTaskClassifier(
            checkpoint, num_binary_outputs=sum(question_counts.values()),
        ).eval()
        question_idxs = {
            'v1': (0, 376),
            'v2': (376, 376 + 142),
            'v3_boostexamples': (376 + 142, 376 + 142 + 156),
        }
        self.question_idxs = question_idxs[qa_questions_version]

        state_dict = torch.load(join(root_dir, 'finetune', f'{checkpoint}.pt'))
        self.model.load_state_dict(state_dict)
        self.model = torch.nn.DataParallel(self.model).to('cuda')

    def get_embs_from_text_list(self, texts: List[str], batch_size=64):
        '''Continuous outputs for each prediction
        '''
        with torch.no_grad():
            inputs = self.tokenizer.batch_encode_plus(
                texts, padding="max_length",
                truncation=True, max_length=512, return_tensors="pt")

            inputs = inputs.to('cuda')

            answer_predictions = []
            for i in tqdm(range(0, len(inputs['input_ids']), batch_size)):
                outputs = self.model(**{k: v[i:i+batch_size]
                                        for k, v in inputs.items()})
                answer_predictions.append(outputs.cpu().detach().numpy())
            answer_predictions = answer_predictions[self.question_idxs[0]
                :self.question_idxs[1]]
            answer_predictions = np.vstack(answer_predictions)
            answer_predictions = scipy.special.softmax(
                answer_predictions, axis=-1)
            return answer_predictions


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
            self.checkpoint = checkpoint
        elif 'Meta-Llama-3' in checkpoint and 'Instruct' in checkpoint:
            if '-refined' in checkpoint:
                self.prompt = '<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nRead the input then answer a question about the input.\n**Input**: "{example}"\n**Question**: {question}\nAnswer with yes or no, then give an explanation.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n**Answer**:'
                self.checkpoint = checkpoint.replace('-refined', '')
            elif '-fewshot' in checkpoint:
                self.prompt = '<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a concise, helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nInput text: and i just kept on laughing because it was so\nQuestion: Does the input mention laughter?\nAnswer with Yes or No.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nYes<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nInput text: what a crazy day things just kept on happening\nQuestion: Is the sentence related to food preparation?\nAnswer with Yes or No.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nNo<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nInput text: i felt like a fly on the wall just waiting for\nQuestion: Does the text use a metaphor or figurative language?\nAnswer with Yes or No.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nYes<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nInput text: he takes too long in there getting the pans from\nQuestion: Is there a reference to sports?\nAnswer with Yes or No.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nNo<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nInput text: was silent and lovely and there was no sound except\nQuestion: Is the sentence expressing confusion or uncertainty?\nAnswer with Yes or No.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nNo<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nInput text: {example}\nQuestion: {question}\nAnswer with Yes or No.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'
                self.checkpoint = checkpoint.replace('-fewshot', '')
            else:
                self.prompt = '<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a concise, helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nInput text: {example}\nQuestion: {question}\nAnswer with yes or no, then give an explanation.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'
                self.checkpoint = checkpoint

            # set batch_size
            if '8B' in checkpoint:
                if 'fewshot' in checkpoint:
                    self.batch_size = 16
                else:
                    self.batch_size = 64
            elif '70B' in checkpoint:
                if 'fewshot' in checkpoint:
                    self.batch_size = 64
                else:
                    self.batch_size = 128  # requires 8 GPUs

        else:
            self.prompt = 'Input: {example}\nQuestion: {question} Answer yes or no.\nAnswer:'

        # self.llm = guidance.models.Transformers("meta-llama/Llama-2-13b-hf")
        # print('PROMPT', self.prompt)
        self.llm = imodelsx.llm.get_llm(
            self.checkpoint, CACHE_DIR=expanduser("~/cache_qa_embedder"))
        # self.llm = LLM(model=checkpoint,
        #    tensor_parallel_size=torch.cuda.device_count())
        # self.sampling_params = SamplingParams(temperature=0, max_tokens=1)
        self.use_cache = use_cache

    def __call__(self, examples: List[str], verbose=True) -> np.ndarray:
        # run in one batch
        programs = [
            self.prompt.format(example=example, question=question)
            for example in examples
            for question in self.questions
        ]
        # outputs = self.llm.generate(
        #     programs,
        #     self.sampling_params,
        # )
        # answers = [outputs[i].outputs[0].text for i in range(len(programs))]

        # answers = self.llm(
        #     programs,
        #     max_new_tokens=1,
        #     verbose=verbose,
        #     use_cache=self.use_cache,
        #     batch_size=self.batch_size,
        # )

        # run in batches
        answers = []
        # pass in this multiple to pipeline, even though it still uses batch_size under the hood
        batch_size_mult = self.batch_size * 8
        for i in tqdm(range(0, len(programs), batch_size_mult)):
            answers += self.llm(
                programs[i:i+batch_size_mult],
                max_new_tokens=1,
                verbose=verbose,
                use_cache=self.use_cache,
                batch_size=self.batch_size,
            )

        # print('answers', answers)
        answers = list(map(lambda x: 'yes' in x.lower(), answers))
        answers = np.array(answers).reshape(len(examples), len(self.questions))
        embeddings = np.array(answers, dtype=float)

        return embeddings


def get_big_test():
    questions = [
        'Is the input related to food preparation?',
        'Does the input mention laughter?',
        'Is there an expression of surprise?',
        'Is there a depiction of a routine or habit?',
        'Is there stuttering or uncertainty in the input?',
        # 'Is there a first-person pronoun in the input?',
    ]
    examples = [
        'i sliced some cucumbers and then moved on to what was next',
        'the kids were giggling about the silly things they did',
        'and i was like whoa that was unexpected',
        'walked down the path like i always did',
        'um no um then it was all clear',
        # 'i was walking to school and then i saw a cat',
    ]
    return questions, examples


if __name__ == "__main__":
    # questions = [
    #     'Is the input related to food preparation?',
    #     'Does the input mention laughter?',
    # ]
    # examples = ['I sliced some cucumbers',
    #             'The kids were laughing', 'walking to school I was']
    questions, examples = get_big_test()
    # checkpoint = 'gpt2'
    # checkpoint = 'mistralai/Mistral-7B-Instruct-v0.2'
    # checkpoint = 'mistralai/Mixtral-8x7B-Instruct-v0.1'
    # checkpoint = "meta-llama/Llama-2-7b-hf"
    # checkpoint = "meta-llama/Llama-2-70b-hf"
    # checkpoint = "mistralai/Mixtral-8x7B-v0.1"
    # checkpoint = 'mistralai/Mistral-7B-v0.1'
    # checkpoint = "meta-llama/Meta-Llama-3-8B"
    # checkpoint = "meta-llama/Meta-Llama-3-8B-Instruct"
    # checkpoint = 'meta-llama/Meta-Llama-3-8B-Instruct-fewshot'
    checkpoint = 'meta-llama/Meta-Llama-3-8B-Instruct-refined'
    # checkpoint = 'meta-llama/Meta-Llama-3-70B-Instruct-refined'

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

    # questions = qa_questions.get_questions(version='v6', full=True)  # [:5]
    # embedder = QuestionEmbedder(questions=questions, checkpoint=checkpoint)
    # embedder = QuestionEmbedder(
    # questions=qa_questions.get_questions(), checkpoint=checkpoint)
    embedder = QuestionEmbedder(
        questions=questions, checkpoint=checkpoint, use_cache=False)
    embeddings = embedder(examples)
    df = pd.DataFrame(embeddings.astype(int), columns=[
                      q.split()[-1] for q in questions])
    print('examples x questions')
    print(df)

    # generate prompt for llama3
    # import transformers
    # messages = [
    #     {"role": "system", "content": "You are a concise, helpful assistant."},
    #     {"role": "user", "content": "Input text: and i just kept on laughing because it was so\nQuestion: Does the input mention laughter?\nAnswer with Yes or No."},
    #     {"role": "assistant",
    #         "content": "Yes"},
    #     {"role": "user", "content": "Input text: what a crazy day things just kept on happening\nQuestion: Is the sentence related to food preparation?\nAnswer with Yes or No."},
    #     {"role": "assistant",
    #         "content": "No"},
    #     {"role": "user", "content": "Input text: i felt like a fly on the wall just waiting for\nQuestion: Does the text use a metaphor or figurative language?\nAnswer with Yes or No."},
    #     {"role": "assistant",
    #         "content": "Yes"},
    #     {"role": "user", "content": "Input text: he takes too long in there getting the pans from\nQuestion: Is there a reference to sports?\nAnswer with Yes or No."},
    #     {"role": "assistant",
    #         "content": "No"},
    #     {"role": "user", "content": "Input text: was silent and lovely and there was no sound except\nQuestion: Is the sentence expressing confusion or uncertainty?\nAnswer with Yes or No."},
    #     {"role": "assistant",
    #         "content": "No"},
    #     {"role": "user", "content": "Input text: {example}\nQuestion: {question}\nAnswer with Yes or No."},
    # ]
    # pipeline = transformers.pipeline(
    #     "text-generation",
    #     model="meta-llama/Meta-Llama-3-8B-Instruct",
    #     model_kwargs={"torch_dtype": torch.bfloat16},
    #     device_map="auto",
    # )
    # prompt = pipeline.tokenizer.apply_chat_template(
    #     messages,
    #     tokenize=False,
    #     add_generation_prompt=True
    # )
    # print(repr(prompt))

    # outputs = pipeline(
    #     prompt,
    #     max_new_tokens=256,
    #     # eos_token_id=terminators,
    #     do_sample=True,
    #     temperature=0.6,
    #     top_p=0.9,
    # )
    # print(repr(outputs[0]["generated_text"]))
